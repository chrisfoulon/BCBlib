# Best Overlap: Bayesian Connectivity Analysis

## Quick Start (For Novice Users)

### What does this tool do?

The `best_overlap` tool identifies brain regions that maintain preserved connectivity despite white matter damage (e.g., from stroke lesions). It compares disconnection probability maps from patients with reference connectivity maps (e.g., from activation likelihood estimation studies or tract-specific disconnectomes) to find regions where connections are likely preserved.

### Basic Usage

```bash
python -m bcblib.scripts.run_best_overlap \
    --patient_disconnectome patient_lesion.nii.gz \
    --cluster_disconnectome cluster1.nii.gz cluster2.nii.gz \
    --parcellation AICHA.nii.gz \
    --output_folder results/
```

### Understanding Your Results

The analysis produces four main output files per patient:

1. **Simplified CSV Report** (`*_analysis_report.csv`): Ranked list of brain regions — key metrics only, designed for direct interpretation.
   - `PosteriorMean` — connectivity score (log-scale); higher = stronger preserved connectivity
   - `ROPE_Category` — `Meaningful`, `Negligible`, or `Inconclusive` (see below)
   - `StatisticalQuality` — model-fit and precision label
   - `RatioOverlap`, `WeightedClusterContribution`, `P90` — key raw features

2. **Full CSV Report** (`*_analysis_report_full.csv`): All intermediate metrics preserved for re-analysis and debugging.

3. **Model Summary** (`*_model_summary.txt`): Statistical diagnostics
   - `Bayesian R²` tells you overall model quality:
     - > 0.7 = Good fit (trust the results)
     - 0.3-0.7 = Moderate fit (results are suggestive)
     - < 0.3 = Poor fit (interpret with caution)
   - Estimated `ν` (degrees of freedom) reports whether heavy-tailed residuals were detected

4. **Visualization** (`*_plot.png`): Graph showing connectivity scores with 95% credible interval error bars

### Interpreting ROPE Categories

ROPE (Region of Practical Equivalence) provides a Bayesian measure of effect size — whether preserved connectivity is meaningfully above a negligible threshold:

- **Meaningful**: The entire 95% credible interval lies above the threshold → a viable target region
- **Negligible**: The entire 95% credible interval lies within the negligible zone → not a meaningful target
- **Inconclusive**: The credible interval straddles the threshold → insufficient data to decide

The ROPE threshold is computed automatically as the **25th percentile of the raw observed log(WCC) distribution** (computed before model fitting, so independent of the posterior), or can be set explicitly with `--rope_high`.

### Interpreting Statistical Quality Categories

- **Strong_High_Confidence**: Best candidates, strong association with low uncertainty
- **Strong_Moderate_Confidence**: Strong association but more uncertain
- **Moderate_High_Confidence**: Moderate association with low uncertainty
- **Moderate_Confidence**: Moderate association with moderate uncertainty
- **Weak_Model_**: Model fit is not strong, interpret cautiously
- **Poor_Model_Fit**: Results unreliable, likely insufficient data

**Important**: These categories reflect *statistical confidence* (model fit × precision), not clinical importance. Always consider the `ROPE_Category` and raw features (`P90`, `RatioOverlap`) for anatomical plausibility.

---

## For Advanced Users

### Input Requirements

#### Disconnectome Maps (NIfTI format)
- **Patient disconnectome**: 3D probability map (0–1) where each voxel indicates probability that a white matter tract passing through that voxel is disconnected by the patient's lesion
- **Cluster disconnectome(s)**: Reference connectivity maps (e.g., from tractography, ALE meta-analysis) representing connections of interest

Maps must:
- Be in the same stereotactic space
- Have identical dimensions and voxel sizes
- Contain probability values (typically 0–1 range)

#### Parcellation Atlas
- Integer-labeled 3D NIfTI image defining brain regions
- Each unique positive integer represents one region
- Must be in the same space as disconnectome maps

### How Preserved Connections Are Computed

The tool computes preserved connection probability using **probability multiplication** (not subtraction):

```
P(preserved) = P(cluster_affected) × [1 - P(patient_disconnected)]
```

**Rationale**: This represents the joint probability that:
1. A connection is vulnerable/affected in the reference map, AND
2. That connection is NOT disconnected by the patient's lesion

This is mathematically correct for independent probabilities and avoids negative values that would result from subtraction.

### Features Extracted for Each Region

For each parcellation region, the following features are computed from the preserved connection map:

| Feature | Description | Range |
|---------|-------------|-------|
| **ClusterVolume** | Total number of voxels in the region | Integer |
| **OverlapVolume** | Number of voxels with preserved connections (> 0) | Integer |
| **SumProb** | Sum of all preserved connection probabilities | Float ≥ 0 |
| **P90** | 90th percentile of preserved probabilities | 0–1 |
| **RatioOverlap** | OverlapVolume / ClusterVolume | 0–1 |
| **DensityOverlap** | SumProb / OverlapVolume (avg. prob where overlap exists) | Float ≥ 0 |
| **WeightedClusterContribution** | SumProb / ClusterVolume | Float ≥ 0 |

**Domain Consideration**: High `RatioOverlap` (e.g., 0.8) does not guarantee importance. A focal preserved connection (low ratio) with high probability (high P90) may be more clinically meaningful depending on the tract anatomy. Raw features should always be examined in anatomical context.

### Optional Detection-Limit Filters

Two complementary optional filters can exclude region-cluster pairs that are below a meaningful spatial coverage threshold **before** model fitting. Both are off by default.

| Parameter | Criterion | Purpose |
|-----------|-----------|---------|
| `--min_ratio_overlap` | `RatioOverlap ≥ threshold` | Exclude pairs where too small a fraction of the region has any preserved connections. Size-normalised: avoids penalising small regions unfairly. |
| `--min_overlap_voxels` | `OverlapVolume ≥ threshold` | Exclude pairs with fewer than N overlap voxels. Guards against single-voxel partial-volume artefacts. |

When both are set, both must be satisfied (AND logic). If more than 20% of pairs are removed, a warning is printed.

> **Warning**: Set these thresholds based on spatial resolution or anatomical grounds, not to improve the statistical results after inspecting the output. Choosing thresholds post-hoc based on what produces better-looking results is equivalent to p-hacking (Simmons et al., 2011). A size-normalised ratio threshold (`--min_ratio_overlap`) guards against the same region-size confound that motivated the `WeightedClusterContribution` outcome variable. A fixed-voxel threshold (`--min_overlap_voxels`) addresses partial-volume artefacts, analogous to minimum cluster size in fMRI (Forman et al., 1995).

---

## Technical Details

### Bayesian Regression Model

#### Outcome Variable

The outcome is `log(WeightedClusterContribution + ε)`, where
`WeightedClusterContribution = SumProb / ClusterVolume` and `ε = 1e-9`.

Normalising `SumProb` by region volume removes a systematic size confound: large atlas regions accumulate high `SumProb` values simply by containing more voxels, not because they are more meaningfully connected. This is standard practice in voxel-based morphometry (Ashburner & Friston, 2000) and lesion-symptom mapping (Rorden et al., 2007).

#### Model Specification

```
y_i = log(WeightedClusterContribution_i + ε)

y_i ~ Student-t(μ_i, σ, ν)

μ_i = β₀ + β_cv·z(log(ClusterVolume_i + 1))
         + β_p90·z(P90_i)
         + β_ro·z(RatioOverlap_i)
         + β_de·z(DensityOverlap_i)

ν ~ Exponential(1/29)
```

where `z(·)` denotes standardisation to zero mean and unit standard deviation.

**Key aspects**:
- Outcome normalised by region volume removes the size confound in raw `SumProb`
- `WeightedClusterContribution` is the **outcome**, not a predictor (including it as both would be a tautology)
- All predictors are z-scored, making coefficients interpretable as Cohen's d-like effect sizes (Cohen, 1988)
- Weakly informative priors: `β ~ Normal(0, 10)` allow data to dominate

#### Robust Likelihood (Student-t)

The Normal likelihood assumes homogeneous residuals. After volume-normalisation, some heteroscedasticity may remain (e.g., high-probability focal connections have different noise structure than diffuse low-probability ones). The Student-t likelihood with estimated degrees-of-freedom ν handles this robustly.

The prior `ν ~ Exponential(1/29)` has mean 29 (close to Normal) while permitting heavy tails if the data demands them (Juárez & Steel, 2010). The model summary reports the estimated ν with interpretation:

- **ν < 5**: Strong evidence of heavy-tailed residuals; Normal likelihood would have been inappropriate
- **ν 5–30**: Moderate departure from Normality; Student-t provides meaningful robustness
- **ν > 30**: Residuals consistent with Normal; Student-t provides robustness at negligible cost

**References**: Lange et al. (1989); Gelman et al., BDA3 (2014), Ch. 17.

#### Why Standardize Predictors?

Standardising (mean = 0, SD = 1) enables:
1. **Coefficients as effect sizes**: |β| > 0.5 = large, > 0.3 = medium, > 0.1 = small (Cohen, 1988)
2. **Direct comparison** of relative importance across predictors
3. **Numerical stability** in MCMC sampling

### ROPE Classification

ROPE (Region of Practical Equivalence) is a Bayesian tool for distinguishing connectivity that is statistically above a meaningful threshold from connectivity that is real but negligible — the Bayesian analog of "statistically significant but trivially small effect size" in frequentist statistics (Kruschke, 2018).

The ROPE is one-sided: `(−∞, rope_high]`. Three outcomes:

| Category | Condition | Interpretation |
|----------|-----------|---------------|
| **Meaningful** | `CI_lower > rope_high` | Connectivity confidently above negligible threshold; candidate target |
| **Negligible** | `CI_upper ≤ rope_high` | Connectivity indistinguishable from background; not a viable target |
| **Inconclusive** | CI straddles boundary | Insufficient data to classify; needs more data or different approach |

The default `rope_high` is the **25th percentile of the raw observed log(WCC) distribution**, computed from the feature data before model fitting (independent of the posterior). It places the boundary at the value below which the bottom quarter of *measured* connectivity scores lie.

> **Important — exploratory vs confirmatory use**: The automatic default produces a *relative* classification ("confidently in the bottom quartile of this patient's measured connectivity"). For any confirmatory or publication use, `rope_high` **must be set explicitly** with a biological justification written before the analysis runs — for example: *"WCC < 0.001 is below our spatial-resolution detection limit, so rope_high = log(0.001) ≈ −6.9."* Do not tune `rope_high` after inspecting the output; doing so is equivalent to p-hacking. The default is provided for exploratory and quality-control purposes only.

**References**: Kruschke (2018); Kruschke & Liddell (2018); Makowski et al. (2019).

### Model Evaluation Metrics

#### Bayesian R² (Gelman et al., 2019)

Proportion of variance explained by the model:

```
R² = Var(fitted) / [Var(fitted) + Var(residual)]
```

Computed from the posterior predictive distribution, accounting for full posterior uncertainty.

**Interpretation thresholds**:
- R² > 0.9: Excellent fit
- R² > 0.7: Good fit
- R² > 0.5: Moderate fit
- R² > 0.3: Weak fit
- R² < 0.3: Poor fit (results unreliable)

#### LOO-CV (Vehtari et al., 2017)

Leave-one-out cross-validation using Pareto-smoothed importance sampling (PSIS-LOO). Assesses out-of-sample predictive accuracy without refitting the model N times.

**Lower LOO values indicate better prediction**. `p_loo` (effective number of parameters) helps detect overfitting.

#### Effect Sizes (Standardised Coefficients)

Because predictors are z-scored, the posterior mean of each β coefficient directly indicates:
- **Magnitude**: |β| quantifies association strength
- **Direction**: sign indicates positive/negative relationship
- **Relative importance**: |β| values are directly comparable across predictors

**Thresholds** (Cohen, 1988):
- |β| > 0.5: Large effect
- |β| > 0.3: Medium effect
- |β| > 0.1: Small effect
- |β| < 0.1: Negligible effect

### Statistical Quality Categories

Categories are assigned hierarchically based on:

1. **Model fit gate**: R² < 0.3 → "Poor_Model_Fit" (stop)
2. **Confidence threshold**: RelativeUncertainty < 0.5 (CI width < 50% of |estimate|)
3. **Strength criterion**: PosteriorMean > median(PosteriorMean)

| Category | R² | Strength | Confidence |
|----------|-----|----------|------------|
| Strong_High_Confidence | > 0.7 | High | High |
| Strong_Moderate_Confidence | > 0.7 | High | Moderate |
| Moderate_High_Confidence | > 0.7 | Moderate | High |
| Moderate_Confidence | > 0.7 | Moderate | Moderate |
| Weak_Model_High_Confidence | 0.3–0.7 | Any | High |
| Weak_Model_Moderate_Confidence | 0.3–0.7 | Any | Moderate |

**Critical Note**: These categories reflect statistical properties (model fit, precision), NOT anatomical or clinical importance. A region with "Moderate_Confidence" but focal high-probability connections (high P90, low RatioOverlap) may be more interesting than one with "Strong_High_Confidence" but diffuse weak connections. Use `ROPE_Category` as a complementary filter.

### Output Files

Per patient, two CSV files are written:

**Simplified CSV** (`*_analysis_report.csv`) — user-facing; designed for direct interpretation:

| Column | Description |
|--------|-------------|
| Target_Region_ID | Parcellation region integer label |
| Source_Cluster | Cluster disconnectome name |
| PosteriorMean | Posterior mean of log(WCC) — connectivity score |
| CredibleInterval_2.5 | Lower bound of 95% credible interval |
| CredibleInterval_97.5 | Upper bound of 95% credible interval |
| ROPE_Category | Meaningful / Negligible / Inconclusive |
| StatisticalQuality | Model-fit × precision category |
| BayesianR2 | Bayesian R² for the model |
| RatioOverlap | Fraction of region voxels with preserved connections |
| WeightedClusterContribution | SumProb / ClusterVolume (size-normalised outcome) |
| P90 | 90th percentile of preserved probabilities |

**Full CSV** (`*_analysis_report_full.csv`) — all intermediate metrics for re-analysis:

In addition to the simplified columns, the full CSV contains `ROPE_High` (the ROPE boundary used), `CI_Width`, `RelativeUncertainty`, `ClusterVolume`, `OverlapVolume`, `SumProb`, and `DensityOverlap`.

If an atlas label file is provided, translated versions of both files are written: `translated_*_analysis_report.csv` and `translated_*_analysis_report_full.csv`.

### Analyzing Multiple Clusters Simultaneously

When multiple cluster disconnectomes are provided, all cluster-region combinations are analyzed in a **single Bayesian model**. This is essential because:

1. **Shared parameters**: Intercept and error variance are estimated jointly
2. **Direct comparability**: Posterior means are on the same scale
3. **Statistical validity**: Avoids multiple comparison issues from separate models

Each region-cluster combination is tagged with `Source_Cluster` in the output CSV.

**Example interpretation**: If cluster A shows higher posterior means for region X than cluster B, region X has stronger preserved connectivity from cluster A's pathways than cluster B's pathways (assuming similar R² values).

---

## Mathematical Foundations

### Probability of Preserved Connections

Given two independent events:
- A: Connection is vulnerable in reference map (prob = p_cluster)
- B: Connection is disconnected by patient lesion (prob = p_patient)

The probability that a connection is both vulnerable AND not disconnected is:

```
P(A ∩ B̄) = P(A) × P(B̄) = p_cluster × (1 - p_patient)
```

**Why multiplication, not subtraction?**
- Subtraction (p_cluster - p_patient) can yield negative values, which are meaningless for probabilities
- Multiplication correctly represents joint probability of independent events
- Preserves probability axioms: 0 ≤ P(preserved) ≤ 1

### Bayesian Inference via MCMC

The tool uses **Hamiltonian Monte Carlo** (HMC) via PyMC to sample from the posterior distribution:

```
P(β, σ, ν | data) ∝ P(data | β, σ, ν) × P(β) × P(σ) × P(ν)
```

**Advantages of Bayesian approach**:
- Full posterior distribution, not just point estimates
- Natural uncertainty quantification via credible intervals
- Probabilistic rankings (posterior means) respect uncertainty
- No p-values or multiple comparison corrections needed

**MCMC Diagnostics**: The tool uses 2000 post-warmup samples with default PyMC settings (4 chains). Convergence should be checked if results seem unexpected (look for warnings in console output).

### Credible Intervals vs. Confidence Intervals

**Credible intervals** (Bayesian): "95% CI [a, b]" means there is 95% probability that the true parameter lies between a and b, given the data and model.

**Confidence intervals** (frequentist): "95% CI [a, b]" means if we repeated the experiment many times, 95% of the intervals would contain the true parameter.

For practical interpretation, Bayesian credible intervals are often more intuitive: they directly quantify uncertainty about parameter values.

---

## References

### Statistical Methods

**Ashburner, J., & Friston, K.J.** (2000). Voxel-based morphometry — the methods. *NeuroImage*, 11(6), 805–821. https://doi.org/10.1006/nimg.2000.0582
- Canonical reference for volume normalisation in neuroimaging statistics (rationale for WeightedClusterContribution)

**Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Foundation for effect size interpretation (small/medium/large)
- Standardised coefficients as effect sizes: pp. 77–81

**Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., & Rubin, D.B.** (2014). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Standard textbook treatment of robust Bayesian regression with Student-t likelihoods: Chapter 17

**Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A.** (2019). R-squared for Bayesian Regression Models. *The American Statistician*, 73(3), 307–309. https://doi.org/10.1080/00031305.2018.1549100
- Bayesian R² computation and interpretation

**Juárez, M.A., & Steel, M.F.J.** (2010). Model-based clustering of non-Gaussian panel data based on skew-t distributions. *Journal of Business & Economic Statistics*, 28(1), 52–66.
- Prior choice `ν ~ Exponential(1/29)`: mode near Normal (29), allows heavy tails

**Kruschke, J.K.** (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science*, 1(2), 270–280. https://doi.org/10.1177/2515245918771304
- Primary ROPE reference: defines the framework, decision rules, and how to set bounds

**Kruschke, J.K., & Liddell, T.M.** (2018). The Bayesian New Statistics: Hypothesis testing, estimation, meta-analysis, and power analysis from a Bayesian perspective. *Psychonomic Bulletin & Review*, 25(1), 178–206. https://doi.org/10.3758/s13423-016-1221-4
- Broader context for ROPE in the Bayesian framework, contrasting with null-hypothesis testing

**Lange, K.L., Little, R.J.A., & Taylor, J.M.G.** (1989). Robust statistical modeling using the t distribution. *Journal of the American Statistical Association*, 84(408), 881–896. https://doi.org/10.1080/01621459.1989.10478852
- Foundational paper establishing the t distribution as a robust alternative for regression with heterogeneous residuals

**Makowski, D., Ben-Shachar, M.S., & Lüdecke, D.** (2019). bayestestR: Describing effects and their uncertainty, existence and significance within the Bayesian framework. *Journal of Open Source Software*, 4(40), 1541. https://doi.org/10.21105/joss.01541
- Practical ROPE implementation reference

**Rorden, C., Karnath, H.-O., & Bonilha, L.** (2007). Improving lesion-symptom mapping. *Journal of Cognitive Neuroscience*, 19(7), 1081–1088.
- Discusses region-size confounds in lesion-overlap analyses (rationale for volume normalisation)

**Simmons, J.P., Nelson, L.D., & Simonsohn, U.** (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science*, 22(11), 1359–1366.
- Rationale for not filtering on outcome-related metrics after seeing results (p-hacking)

**Vehtari, A., Gelman, A., & Gabry, J.** (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432. https://doi.org/10.1007/s11222-016-9696-4
- LOO-CV methodology and PSIS diagnostic

### Neuroimaging Applications

**Foulon, C., Cerliani, L., Kinkingnéhun, S., et al.** (2018). Advanced lesion symptom mapping analyses and implementation as BCBtoolkit. *GigaScience*, 7(3), giy004. https://doi.org/10.1093/gigascience/giy004
- BCBtoolkit methodology and disconnectome computation

**Thiebaut de Schotten, M., & Forkel, S.J.** (2022). The emergent properties of the connected brain. *Science*, 378(6619), 505–510. https://doi.org/10.1126/science.abq2591
- Conceptual framework for preserved connectivity analysis

### Probability Theory

**Kolmogorov, A.N.** (1933/1950). *Foundations of the Theory of Probability* (English translation). Chelsea Publishing Company.
- Multiplication rule for independent events: Chapter 1, Section 4

---

## How to Cite the Statistical Methods

When using this tool in a publication, please cite:

**For the tool itself**:
> Foulon, C., Cerliani, L., Kinkingnéhun, S., et al. (2018). Advanced lesion symptom mapping analyses and implementation as BCBtoolkit. *GigaScience*, 7(3), giy004. https://doi.org/10.1093/gigascience/giy004

**For the volume-normalised outcome (WeightedClusterContribution)**:
> Ashburner, J., & Friston, K.J. (2000). Voxel-based morphometry — the methods. *NeuroImage*, 11(6), 805–821.
> Rorden, C., Karnath, H.-O., & Bonilha, L. (2007). Improving lesion-symptom mapping. *Journal of Cognitive Neuroscience*, 19(7), 1081–1088.

**For the Student-t likelihood (robust regression)**:
> Lange, K.L., Little, R.J.A., & Taylor, J.M.G. (1989). Robust statistical modeling using the t distribution. *JASA*, 84(408), 881–896.
> Gelman, A., et al. (2014). *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 17.

**For ROPE classification**:
> Kruschke, J.K. (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science*, 1(2), 270–280. https://doi.org/10.1177/2515245918771304

**For Bayesian R²**:
> Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019). R-squared for Bayesian Regression Models. *The American Statistician*, 73(3), 307–309.

**For LOO-CV**:
> Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.

---

## Example Workflow

### Complete Analysis with Commissural Connections

```bash
# Run analysis for one patient with two ALE clusters
python -m bcblib.scripts.run_best_overlap \
    --patient_disconnectome patient001_lesion_disco.nii.gz \
    --cluster_disconnectome semantics_ALE_cluster.nii.gz phonology_ALE_cluster.nii.gz \
    --parcellation AICHA_MNI.nii.gz \
    --atlas_labels AICHA_labels.txt \
    --output_folder results/patient001/

# Output files:
# - patient001_analysis_report.csv          (simplified, user-facing)
# - patient001_analysis_report_full.csv     (all metrics, for re-analysis)
# - patient001_analysis_report_model_summary.txt
# - patient001_analysis_report_plot.png
# - translated_patient001_analysis_report.csv
# - translated_patient001_analysis_report_full.csv
# - derivatives/patient001/*_preserved.nii.gz
```

### Using Detection-Limit Filters

```bash
# Require at least 2% of region to have any preserved connections,
# AND at least 5 overlap voxels (2 mm isotropic data quality floor)
python -m bcblib.scripts.run_best_overlap \
    --patient_disconnectome patient001.nii.gz \
    --cluster_disconnectome cluster.nii.gz \
    --parcellation AICHA.nii.gz \
    --output_folder results/ \
    --min_ratio_overlap 0.02 \
    --min_overlap_voxels 5
```

> **Reminder**: choose these thresholds before inspecting results, based on anatomy and spatial resolution.

### Interpreting Results

1. **Check model quality**: Open `*_model_summary.txt`, look at Bayesian R² and estimated ν
   - If R² < 0.3, results are unreliable
   - If ν < 5, there are heavy-tailed residuals (the Student-t likelihood handled this)

2. **Examine effect sizes**: Which features drive connectivity predictions?
   - Large |β| for `log_ClusterVolume` means region size matters even after normalisation
   - Large |β| for `P90` means high peak probabilities drive the predictions

3. **Review top candidates**: Open simplified CSV, sort by `PosteriorMean`
   - Focus on `ROPE_Category = "Meaningful"` first
   - Within Meaningful, use `StatisticalQuality` and `P90` for further prioritisation

4. **Compare across clusters**: If analyzing multiple clusters:
   - Same model applies to all (single fit)
   - Compare `PosteriorMean` across `Source_Cluster` for the same `Target_Region_ID`

5. **Visualize**: Load preserved connection images in neuroimaging software
   - Overlay on patient's anatomical MRI
   - Verify anatomical plausibility of high-ranking regions

---

## Frequently Asked Questions

**Q: Can I use this tool with single voxels instead of parcellation regions?**
A: Not directly. The tool requires a parcellation atlas to group voxels into meaningful anatomical units. Voxel-wise analysis would require a different statistical approach (e.g., mass-univariate regression with multiple comparison correction).

**Q: What if my Bayesian R² is low (< 0.3)?**
A: This suggests the features don't explain much variance in preserved connectivity. Possible causes: (1) insufficient anatomical specificity in disconnectome maps, (2) parcellation regions too coarse/fine, (3) lesion doesn't affect connections of interest. Consider examining preserved connection images visually.

**Q: Why are my credible intervals so wide?**
A: Wide intervals (high `RelativeUncertainty`) indicate substantial uncertainty, often due to: (1) small overlap volumes, (2) high variability in preserved probabilities within a region, (3) limited data. This is honest uncertainty quantification — the model is telling you it is not confident.

**Q: What about negative PosteriorMean values?**
A: Negative values are possible because the outcome is log-transformed (`log(WCC + ε)`). Very small WCC values (near zero) produce large negative log-values. Such pairs should have `ROPE_Category = "Negligible"` — use this column rather than the sign of PosteriorMean to interpret very low connectivity.

**Q: How do I choose which regions to target for therapy?**
A: This tool provides statistical rankings of preserved connectivity strength. Clinical decisions require: (1) integration with behavioural/imaging data, (2) consideration of anatomical accessibility, (3) evaluation of treatment feasibility, (4) clinical expertise. Use the rankings to prioritise candidates for closer examination.

**Q: Can I compare results across different analysis runs?**
A: **No** — do not compare posterior means from separate model runs. Only results from the same model fit are directly comparable. If you need to compare different cluster sets, include all clusters in a single run.

**Q: What does the estimated ν tell me?**
A: ν is the degrees-of-freedom parameter of the Student-t likelihood. Low ν (< 5) means the data had heavy-tailed residuals that a Normal likelihood would have handled poorly — the Student-t was the right choice. High ν (> 30) means the data were approximately Normal and using Student-t costs nothing.

---

## Troubleshooting

**Problem**: "No preserved connections were found"
- **Solution**: Check that disconnectome maps have overlapping non-zero voxels. Verify all inputs are in the same stereotactic space.

**Problem**: LOO-CV fails with numerical errors
- **Solution**: Usually harmless. The model completes but LOO results are unavailable. If concerning, check for extreme values in features or very sparse preserved connection maps.

**Problem**: MCMC divergences or convergence warnings
- **Solution**: Indicates sampling difficulties, often from multicollinearity or extreme feature values. Check correlation between features. Consider removing highly correlated predictors.

**Problem**: All regions have the same StatisticalQuality category
- **Solution**: If all regions have very similar posterior means or uncertainties, categories may collapse. Focus on the continuous `PosteriorMean` ranking and `ROPE_Category` instead.

**Problem**: Detection-limit filter removes all pairs
- **Solution**: The thresholds may be too aggressive for this dataset. Lower `--min_ratio_overlap` or `--min_overlap_voxels`, or remove them entirely. Ensure the values were chosen on anatomical grounds, not to improve results.

---

## Citation

If you use this tool in your research, please cite:

```
Foulon, C., Cerliani, L., Kinkingnéhun, S., et al. (2018).
Advanced lesion symptom mapping analyses and implementation as BCBtoolkit.
GigaScience, 7(3), giy004. https://doi.org/10.1093/gigascience/giy004
```

For the statistical methods, see the "How to Cite" section above.
