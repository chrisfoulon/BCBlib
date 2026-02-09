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

The analysis produces three main output files:

1. **CSV Report** (`*_analysis_report.csv`): Ranked list of brain regions
   - Look at `StatisticalQuality` column for interpretation guidance
   - Higher `PosteriorMean` = stronger preserved connectivity
   - Lower `RelativeUncertainty` = more confident estimate

2. **Model Summary** (`*_model_summary.txt`): Statistical diagnostics
   - `Bayesian R²` tells you overall model quality:
     - > 0.7 = Good fit (trust the results)
     - 0.3-0.7 = Moderate fit (results are suggestive)
     - < 0.3 = Poor fit (interpret with caution)

3. **Visualization** (`*_plot.png`): Graph showing connectivity scores with error bars

### Interpreting Statistical Quality Categories

- **Strong_High_Confidence**: Best candidates, strong association with low uncertainty
- **Strong_Moderate_Confidence**: Strong association but more uncertain
- **Moderate_High_Confidence**: Moderate association with low uncertainty
- **Moderate_Confidence**: Moderate association with moderate uncertainty
- **Weak_Model_**: Model fit is not strong, interpret cautiously
- **Poor_Model_Fit**: Results unreliable, likely insufficient data

**Important**: These categories reflect *statistical confidence*, not clinical importance. Always consider anatomical context using the raw features (P90, RatioOverlap, etc.).

---

## For Advanced Users

### Input Requirements

#### Disconnectome Maps (NIfTI format)
- **Patient disconnectome**: 3D probability map (0-1) where each voxel indicates probability that a white matter tract passing through that voxel is disconnected by the patient's lesion
- **Cluster disconnectome(s)**: Reference connectivity maps (e.g., from tractography, ALE meta-analysis) representing connections of interest

Maps must:
- Be in the same stereotactic space
- Have identical dimensions and voxel sizes
- Contain probability values (typically 0-1 range)

#### Parcellation Atlas
- Integer-labeled 3D NIfTI image defining brain regions
- Each unique positive integer represents one region
- Must be in same space as disconnectome maps

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
| **P90** | 90th percentile of preserved probabilities | 0-1 |
| **RatioOverlap** | OverlapVolume / ClusterVolume | 0-1 |
| **DensityOverlap** | SumProb / OverlapVolume (avg. prob where overlap exists) | Float ≥ 0 |
| **WeightedClusterContribution** | SumProb / ClusterVolume | Float ≥ 0 |

**Domain Consideration**: High `RatioOverlap` (e.g., 0.8) doesn't guarantee importance. A focal preserved connection (low ratio) with high probability (high P90) may be more clinically meaningful depending on the tract anatomy. The raw features should always be examined in anatomical context.

---

## Technical Details

### Bayesian Regression Model

#### Model Specification

The connectivity score is modeled as:

```
y_i = log(SumProb_i + 1)

y_i ~ Normal(μ_i, σ)

μ_i = β₀ + β_cv·log(ClusterVolume_i + 1) + β_p90·P90_i + 
      β_ro·RatioOverlap_i + β_de·DensityOverlap_i + 
      β_wc·WeightedClusterContribution_i
```

**Key aspects**:
- Log-transformation of outcome handles right-skewed distributions and provides multiplicative interpretation
- All predictors are **standardized (z-scored)** before modeling, making coefficients directly interpretable as effect sizes
- Weakly informative priors: β ~ Normal(0, 10) allow data to dominate while preventing extreme values

#### Why Standardize Predictors?

Standardizing (mean=0, SD=1) enables:
1. **Coefficients as effect sizes**: |β| > 0.5 = large, > 0.3 = medium, > 0.1 = small (Cohen, 1988)
2. **Direct comparison** of relative importance across predictors
3. **Numerical stability** in MCMC sampling

### Model Evaluation Metrics

#### Bayesian R² (Gelman et al., 2019)

Proportion of variance explained by the model:

```
R² = Var(fitted) / [Var(fitted) + Var(residual)]
```

Computed from posterior predictive distribution, accounting for full posterior uncertainty.

**Interpretation thresholds**:
- R² > 0.9: Excellent fit
- R² > 0.7: Good fit 
- R² > 0.5: Moderate fit
- R² > 0.3: Weak fit
- R² < 0.3: Poor fit (results unreliable)

#### LOO-CV (Vehtari et al., 2017)

Leave-one-out cross-validation using Pareto-smoothed importance sampling (PSIS-LOO). Assesses out-of-sample predictive accuracy without refitting model N times.

**Lower LOO values indicate better prediction**. `p_loo` (effective number of parameters) helps detect overfitting.

#### Effect Sizes (Standardized Coefficients)

Because predictors are standardized, the posterior mean of each β coefficient directly indicates:
- **Magnitude**: |β| quantifies association strength (Cohen's d-like interpretation)
- **Direction**: Sign indicates positive/negative relationship
- **Relative importance**: Can compare |β| values across predictors

**Thresholds** (Cohen, 1988):
- |β| > 0.5: Large effect
- |β| > 0.3: Medium effect  
- |β| > 0.1: Small effect
- |β| < 0.1: Negligible effect

### Statistical Quality Categories

Categories are assigned hierarchically based on:

1. **Model fit gate**: R² < 0.3 → "Poor_Model_Fit" (stop)
2. **Confidence threshold**: RelativeUncertainty < 0.5 (CI width < 50% of estimate)
3. **Strength criterion**: PosteriorMean > median(PosteriorMean)

| Category | R² | Strength | Confidence |
|----------|-----|----------|------------|
| Strong_High_Confidence | > 0.7 | High | High |
| Strong_Moderate_Confidence | > 0.7 | High | Moderate |
| Moderate_High_Confidence | > 0.7 | Moderate | High |
| Moderate_Confidence | > 0.7 | Moderate | Moderate |
| Weak_Model_High_Confidence | 0.3-0.7 | Any | High |
| Weak_Model_Moderate_Confidence | 0.3-0.7 | Any | Moderate |

**Critical Note**: These categories reflect statistical properties (model fit, precision), NOT anatomical or clinical importance. A region with "Moderate_Confidence" but focal high-probability connections (high P90, low RatioOverlap) may be more interesting than one with "Strong_High_Confidence" but diffuse weak connections.

### Analyzing Multiple Clusters Simultaneously

When multiple cluster disconnectomes are provided, all cluster-region combinations are analyzed in a **single Bayesian model**. This is essential because:

1. **Shared parameters**: Intercept and error variance are estimated jointly
2. **Direct comparability**: Posterior means are on the same scale
3. **Statistical validity**: Avoids multiple comparison issues from separate models

Each region-cluster combination is tagged with `Source_Cluster` in the output CSV, allowing comparison of how different connectivity patterns (clusters) relate to the same set of target regions.

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
P(β, σ | data) ∝ P(data | β, σ) × P(β) × P(σ)
```

**Advantages of Bayesian approach**:
- Full posterior distribution, not just point estimates
- Natural uncertainty quantification via credible intervals
- Probabilistic rankings (posterior means) respect uncertainty
- No p-values or multiple comparison corrections needed
- Hierarchical structure naturally handles multiple clusters

**MCMC Diagnostics**: The tool uses 2000 post-warmup samples with default PyMC settings (4 chains). Convergence should be checked if results seem unexpected (look for warnings in console output).

### Credible Intervals vs. Confidence Intervals

**Credible intervals** (Bayesian): "95% CI [a, b]" means there is 95% probability that the true parameter lies between a and b, given the data and model.

**Confidence intervals** (frequentist): "95% CI [a, b]" means if we repeated the experiment many times, 95% of the intervals would contain the true parameter.

For practical interpretation, Bayesian credible intervals are often more intuitive: they directly quantify uncertainty about parameter values.

---

## References

### Statistical Methods

**Cohen, J.** (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.
- Foundation for effect size interpretation (small/medium/large)
- Standardized coefficients as effect sizes: pp. 77-81

**Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A.** (2019). R-squared for Bayesian Regression Models. *The American Statistician*, 73(3), 307-309. 
- Bayesian R² computation and interpretation
- DOI: [10.1080/00031305.2018.1549100](https://doi.org/10.1080/00031305.2018.1549100)

**Vehtari, A., Gelman, A., & Gabry, J.** (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- LOO-CV methodology and PSIS diagnostic
- DOI: [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4)

**Kruschke, J. K.** (2014). Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan (2nd ed.). Academic Press.
- Comprehensive Bayesian regression tutorial
- Credible interval interpretation: Chapter 25

### Neuroimaging Applications

**Foulon, C., Cerliani, L., Kinkingnéhun, S., et al.** (2018). Advanced lesion symptom mapping analyses and implementation as BCBtoolkit. *GigaScience*, 7(3), giy004.
- BCBtoolkit methodology and disconnectome computation
- DOI: [10.1093/gigascience/giy004](https://doi.org/10.1093/gigascience/giy004)

**Thiebaut de Schotten, M., & Forkel, S. J.** (2022). The emergent properties of the connected brain. *Science*, 378(6619), 505-510.
- Conceptual framework for preserved connectivity analysis
- DOI: [10.1126/science.abq2591](https://doi.org/10.1126/science.abq2591)

### Probability Theory

**Kolmogorov, A. N.** (1933/1950). Foundations of the Theory of Probability (English translation). Chelsea Publishing Company.
- Multiplication rule for independent events: Chapter 1, Section 4
- Axiomatic probability foundations

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
# - patient001_analysis_report.csv (main results)
# - patient001_analysis_report_model_summary.txt (diagnostics)
# - patient001_analysis_report_plot.png (visualization)
# - translated_patient001_analysis_report.csv (with anatomical names)
# - derivatives/patient001/patient001_semantics_ALE_cluster_preserved.nii.gz
# - derivatives/patient001/patient001_phonology_ALE_cluster_preserved.nii.gz
```

### Interpreting Results

1. **Check model quality**: Open `*_model_summary.txt`, look at Bayesian R²
   - If R² < 0.3, results are unreliable

2. **Examine effect sizes**: Which features drive connectivity predictions?
   - Large |β| indicates that feature strongly predicts connectivity
   - Helps understand what anatomical properties matter

3. **Review top candidates**: Open CSV, sort by `PosteriorMean`
   - Focus on `StatisticalQuality` = Strong_High_Confidence
   - But also examine raw features (P90, RatioOverlap) for anatomical plausibility

4. **Compare across clusters**: If analyzing multiple clusters:
   - Same R² applies to all (single model)
   - Compare `PosteriorMean` across `Source_Cluster` for same `Target_Region_ID`
   - Identifies which pathways have strongest preserved connectivity to each region

5. **Visualize**: Load preserved connection images in neuroimaging software
   - Overlay on patient's anatomical MRI
   - Verify anatomical plausibility of high-ranking regions

---

## Frequently Asked Questions

**Q: Can I use this tool with single voxels instead of parcellation regions?**
A: Not directly. The tool requires a parcellation atlas to group voxels into meaningful anatomical units. Voxel-wise analysis would require a different statistical approach (e.g., mass-univariate regression with multiple comparison correction).

**Q: What if my Bayesian R² is low (< 0.3)?**
A: This suggests the features don't explain much variance in preserved connectivity. Possible causes: (1) Insufficient anatomical specificity in disconnectome maps, (2) Parcellation regions too coarse/fine, (3) Lesion doesn't affect connections of interest. Consider examining preserved connection images visually to understand the pattern.

**Q: Why are my credible intervals so wide?**
A: Wide intervals (high `RelativeUncertainty`) indicate substantial uncertainty, often due to: (1) Small overlap volumes, (2) High variability in preserved probabilities within region, (3) Limited data. This is honest uncertainty quantification - the model is telling you it's not confident.

**Q: How do I choose which regions to target for therapy?**
A: This tool provides **statistical rankings of preserved connectivity strength**. Clinical decisions require: (1) Integration with behavioral/imaging data, (2) Consideration of anatomical accessibility, (3) Evaluation of treatment feasibility, (4) Clinical expertise. Use the rankings to prioritize candidates for closer examination, not as sole decision criterion.

**Q: Can I compare results across different analysis runs?**
A: **NO** - do not compare posterior means from separate model runs. Only results from the same model fit are directly comparable. If you need to compare different cluster sets, include all clusters in a single analysis run.

**Q: What about negative PosteriorMean values?**
A: Negative values are possible because the outcome is log-transformed. They indicate regions with very low preserved connectivity (close to zero). The ranking is still valid - higher values indicate stronger connectivity regardless of sign.

---

## Troubleshooting

**Problem**: "No preserved connections were found"
- **Solution**: Check that disconnectome maps have overlapping non-zero voxels. Verify all inputs are in the same stereotactic space.

**Problem**: LOO-CV fails with numerical errors  
- **Solution**: This is usually harmless. The model will complete but LOO results won't be available. If concerning, check for extreme values in features or very sparse preserved connection maps.

**Problem**: MCMC divergences or convergence warnings
- **Solution**: Indicates sampling difficulties, often from multicollinearity or extreme feature values. Check correlation between features. Consider removing highly correlated predictors or increasing target_accept parameter.

**Problem**: All regions have same StatisticalQuality category
- **Solution**: If all regions have very similar posterior means or uncertainties, categories may collapse. This is fine - focus on the continuous `PosteriorMean` ranking instead.

---

## Citation

If you use this tool in your research, please cite:

```
Foulon, C., Cerliani, L., Kinkingnéhun, S., et al. (2018). 
Advanced lesion symptom mapping analyses and implementation as BCBtoolkit. 
GigaScience, 7(3), giy004. https://doi.org/10.1093/gigascience/giy004
```

For the statistical methods:

```
Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019). 
R-squared for Bayesian Regression Models. 
The American Statistician, 73(3), 307-309.
```
