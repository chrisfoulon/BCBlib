# Context: Robust Statistics for best_overlap

*LAD Phase 1 — Context & Planning*
*Feature slug*: `best_overlap_robust_stats`

---

## Level 1 — Plain English Summary

`best_overlap.py` identifies brain regions with preserved white-matter connectivity in individual patients. A Bayesian linear regression model currently predicts `log(SumProb + 1)` from five overlap features. Empirical analysis of real patient data revealed three linked problems:

1. **Bimodal outcome distribution**: `SumProb` spans five orders of magnitude (0.01–251). A small number of large parcellation regions with near-complete cluster overlap dominate the regression and push predictions for sparse-overlap pairs below zero — impossible by construction.
2. **Region-size confound**: Large anatomical regions accumulate high `SumProb` simply because they contain more voxels, not because they are more meaningfully connected. This inflates leverage and creates misleading rankings.
3. **Patient-insensitive output**: In practice, 91 % of region-cluster pairs show < 0.1 variation in posterior mean across patients. The existing `StatisticalQuality` labels are consequently misleading — they describe within-model rank, not absolute biological relevance.

Three changes address this without excluding data or post-hoc manipulation:

- **Fix 1**: Change the outcome from `log(SumProb + 1)` to `log(WeightedClusterContribution + ε)`. `WeightedClusterContribution = SumProb / ClusterVolume` normalises by region size. This is the primary fix — it eliminates the bimodality problem at its root.
- **Fix 2**: Replace the Normal likelihood with a Student-t likelihood (estimated degrees-of-freedom `ν`). This makes the model robust to residual heteroscedasticity after Fix 1.
- **Fix 4**: Add ROPE (Region of Practical Equivalence) classification to the output. This is the Bayesian analog of "real but negligible effect size" — it identifies pairs whose 95 % credible interval falls entirely within a pre-specified negligible range.

A fourth option (Fix 3: hard pre-filtering by minimum overlap) was explicitly evaluated and **rejected as the primary mechanism** because it is post-hoc data selection and not meaningfully distinct from p-hacking unless the threshold is pre-specified on biological grounds independently of the results. It may be offered as an optional user-facing parameter (`--min_overlap_voxels`) documenting it as a detection-limit criterion, not a statistical filter.

---

## Level 2 — API Table (Current State)

| Symbol | Purpose | Key Inputs | Key Outputs | Notes |
|--------|---------|-----------|-------------|-------|
| `compute_cluster_features` | Extract overlap stats per region | `preserved_map`, `parcellation_path` | DataFrame: 8 feature columns | Filters `OverlapVolume == 0` rows |
| `run_bayesian_model` | Fit Bayesian regression | DataFrame with features | `dict`: trace, R², LOO, effect_sizes, std_params | Outcome: `log(SumProb+1)`, likelihood: Normal |
| `_compute_statistical_quality` | Assign quality label | posterior_mean, rel_uncertainty, R², median | str category | Based on R², above-median, CI-width |
| `generate_report` | Build CSV + plot | DataFrame, model_results, path | CSV, PNG, txt summary | Adds Posterior* columns, quality labels |
| `translate_cluster_labels` | Map IDs → names | atlas_txt, input_csv, output_csv | translated CSV | Reads `nom_c`/`color` columns |
| `run_connectivity_analysis` | End-to-end pipeline | patient paths, cluster paths, parcellation | output folder contents | Calls all above; loops over patients |

---

## Level 3 — Key Integration Points

### What changes in `run_bayesian_model`

```python
# Current (problematic)
y = np.log(df["SumProb"] + 1)           # range [0, ~5.5], bimodal
obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)   # sensitive to outliers

# After Fix 1 + Fix 2
EPS = 1e-6  # prevent log(0)
y = np.log(df["WeightedClusterContribution"] + EPS)      # range more uniform
nu = pm.Exponential("nu", lam=1/29)     # weakly informative; mode ~29, allows heavy tails
obs = pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=y)
```

### What changes in `generate_report`

```python
# Fix 4: new ROPE classification function
def compute_rope_classification(ci_lower, ci_upper, rope_low, rope_high):
    ...  # see plan for spec

# New simplified CSV output
SIMPLIFIED_COLS = [
    "Target_Region_ID", "Target_Region_Name",  # if atlas provided
    "Source_Cluster",
    "PosteriorMean", "CredibleInterval_2.5", "CredibleInterval_97.5",
    "ROPE_Category",                           # new
    "RatioOverlap", "WeightedClusterContribution",  # key raw features
    "StatisticalQuality",
]
```

### Outcome variable change effect on downstream columns

`PosteriorMean`, `CredibleInterval_*`, `CI_Width`, `RelativeUncertainty` all come from `mu_det` in the trace, which is now predicting `log(WCC + ε)`. Their scale and interpretation change — the documentation must be updated accordingly.

---

## Maintenance Opportunities in Target Files

### High Priority (address during implementation)

- [ ] `best_overlap.py:571` — `RelativeUncertainty = ci_width / np.abs(posterior_mean)` blows up when `posterior_mean ≈ 0` (currently masked by the large-PM outliers). After Fix 1 this will be more exposed. Add `np.where(np.abs(posterior_mean) < 1e-9, np.inf, ci_width / np.abs(posterior_mean))`.
- [ ] `best_overlap.py:586` — `generate_model_summary` writes `\\n` literal strings (escaped newline) instead of actual newlines in the output file. Bug introduced by double-escaping.
- [ ] `generate_report` currently modifies `df` in-place (`df["PosteriorMean"] = ...`). This is a side-effect that could corrupt the caller's dataframe. Use `.copy()` at function entry.

### Medium Priority (Boy Scout Rule)

- [ ] `best_overlap.py` docstring module header still says "formerly connectivity" — update to reflect current name.
- [ ] Column name `ClusterVolume` is misleading: it is the parcellation region volume, not the disconnectome cluster volume. Rename to `RegionVolume` or document clearly.

---

## Literature References for Non-Standard Choices

These are choices that differ from the default in analogous studies and require justification in documentation and citations in papers.

### Fix 1 — Outcome normalisation by region volume

Normalising a spatial sum by region size is standard practice in voxel-based morphometry and lesion-symptom mapping but is not the default in connectivity analyses. Justification:

- **Ashburner, J., & Friston, K. J.** (2000). Voxel-based morphometry — the methods. *NeuroImage*, 11(6), 805–821. https://doi.org/10.1006/nimg.2000.0582
  — Canonical reference for the rationale of volume normalisation in neuroimaging statistics.
- **Rorden, C., Karnath, H.-O., & Bonilha, L.** (2007). Improving lesion-symptom mapping. *Journal of Cognitive Neuroscience*, 19(7), 1081–1088.
  — Discusses region-size confounds in lesion-overlap analyses.

### Fix 2 — Student-t likelihood (robust regression)

Standard Normal likelihood is the default in Bayesian regression. Using Student-t is non-standard and requires justification:

- **Lange, K. L., Little, R. J. A., & Taylor, J. M. G.** (1989). Robust statistical modeling using the t distribution. *Journal of the American Statistical Association*, 84(408), 881–896. https://doi.org/10.1080/01621459.1989.10478852
  — Foundational paper establishing the t distribution as a robust alternative for regression with heterogeneous residuals.
- **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B.** (2014). *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 17.
  — Standard textbook treatment of robust Bayesian regression with t likelihoods.
- **Juárez, M. A., & Steel, M. F. J.** (2010). Model-based clustering of non-Gaussian panel data based on skew-t distributions. *Journal of Business & Economic Statistics*, 28(1), 52–66.
  — Prior choice for ν: `Exponential(1/29)` puts the mode at 29 (close to Normal) while allowing heavy tails if the data demands them. Widely used default in PyMC community.

### Fix 4 — ROPE (Region of Practical Equivalence)

ROPE is not a frequentist concept. It has no direct analog in classical statistics and requires explanation in any paper using it:

- **Kruschke, J. K.** (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science*, 1(2), 270–280. https://doi.org/10.1177/2515245918771304
  — Primary ROPE reference: defines the framework, decision rules, and how to set bounds.
- **Kruschke, J. K., & Liddell, T. M.** (2018). The Bayesian New Statistics: Hypothesis testing, estimation, meta-analysis, and power analysis from a Bayesian perspective. *Psychonomic Bulletin & Review*, 25(1), 178–206. https://doi.org/10.3758/s13423-016-1221-4
  — Broader context for ROPE in the Bayesian framework, contrasting with null-hypothesis testing.
- **Makowski, D., Ben-Shachar, M. S., & Lüdecke, D.** (2019). bayestestR: Describing effects and their uncertainty, existence and significance within the Bayesian framework. *Journal of Open Source Software*, 4(40), 1541. https://doi.org/10.21105/joss.01541
  — Practical ROPE implementation reference; widely cited when using ROPE in applied work.

### Fix 3 (optional detection-limit filters) — design rationale

Two complementary optional parameters (`--min_ratio_overlap`, `--min_overlap_voxels`) are offered, both off by default. The design choice of `RatioOverlap` (not `WeightedClusterContribution`) as the ratio-filter criterion is deliberate:

- `WeightedClusterContribution` is the regression *outcome* after Fix 1. Filtering on it before fitting sets a minimum response variable value — a direct analogue of p-hacking (**Simmons, J. P., Nelson, L. D., & Simonsohn, U.**, 2011, *Psychological Science*, 22(11), 1359–1366).
- `RatioOverlap` is a *predictor* representing spatial coverage (fraction of region voxels with any preserved connection). Filtering on it is a data-quality / anatomical coverage criterion, not a signal-strength criterion — analogous to minimum read depth in RNA-seq or minimum cluster size in fMRI (**Forman, S. D., et al.**, 1995, *Magnetic Resonance in Medicine*, 33(5), 636–647).
- The size-normalised ratio filter directly addresses the same region-size confound as Fix 1 but at the inclusion stage rather than in the outcome scale.
- The documentation will explicitly warn against choosing thresholds based on whether results look better — both values must be justified on spatial resolution / anatomical grounds and pre-specified before analysis.

### Dual CSV output rationale

No single reference — this follows neuroimaging tool conventions (e.g., FSL randomise produces both raw t-stats and corrected p-maps; FreeSurfer aparc produces both raw thickness and normalised stats). The rationale is traceability: the raw CSV preserves all intermediate quantities for re-analysis; the simplified CSV is for direct clinical/scientific interpretation.

---

## Open Questions for User Before Implementation

1. **ROPE bounds**: What constitutes "negligible preserved connectivity" in your domain? Options:
   - Fixed in log(WCC) space: e.g., `[-log(2), log(2)]` = WCC within factor of 2 of zero
   - Data-driven: ±0.1 SD of the outcome distribution (Kruschke 2018 default)
   - User-specified parameter with a documented default
   We recommend a **user-specifiable parameter with default = ±0.1 SD** as this is both principled and transparent.

2. **Simplified vs raw CSV**: Two options:
   - **Two files**: `*_analysis_report.csv` (simplified, user-facing) + `*_analysis_report_full.csv` (all metrics). Cleaner UX, consistent with neuroimaging conventions.
   - **One file with column grouping**: Single CSV with all columns, but documentation separates them into tiers. Simpler file management, less confusing for scripting.
   Recommendation: **two files**, because the simplified one is what 90 % of users need and the full one is for reanalysis/debugging.

3. **`--min_overlap_voxels` optional parameter**: Should it default to `None` (off) or to a sensible detection limit (e.g., 5 voxels for 2 mm isotropic data)? If a default is set, it must be justified in the documentation as a spatial resolution argument, not a statistical one.
