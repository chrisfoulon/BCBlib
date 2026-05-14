"""
Module: best_overlap
Description: Provides functions to process disconnectome maps by computing preserved
             connections (cluster_prob × (1 − patient_prob)), extracting size-normalised
             connectivity features per parcellation region, fitting a robust Bayesian
             regression model (Student-t likelihood) on the
             WeightedClusterContribution outcome, and generating ranked CSV reports with
             ROPE-based interpretation aids.  Multiple cluster disconnectomes are
             analysed in a single model so that posterior scores are directly comparable
             across clusters and regions.  Optionally, region IDs are translated to
             anatomical names via a label file.

Statistical approach
--------------------
- Outcome: log(WeightedClusterContribution + ε), where
  WeightedClusterContribution = SumProb / ClusterVolume.  Normalising by region
  volume removes the size confound that inflates SumProb for large atlas regions.
- Likelihood: Student-t with estimated degrees-of-freedom ν (Lange et al., 1989;
  Gelman et al., 2014).  ν is inferred from data; low ν signals heavy-tailed
  residuals (heterogeneous data); ν → ∞ recovers the Normal.
- Predictors: log(ClusterVolume+1), P90, RatioOverlap, DensityOverlap — all
  standardised to z-scores so coefficients are interpretable as effect sizes
  (Cohen, 1988).
- Interpretation aid: ROPE (Region of Practical Equivalence) classification
  (Kruschke, 2018) flags pairs whose credible interval lies entirely within a
  negligible range, providing a Bayesian analog of "real but trivially small
  effect size".

References
----------
Cohen, J. (1988). Statistical Power Analysis for the Behavioural Sciences (2nd ed.).
Gelman et al. (2014). Bayesian Data Analysis (3rd ed.), Ch. 17.
Gelman et al. (2019). R-squared for Bayesian Regression Models.
  The American Statistician, 73(3), 307–309.
Kruschke, J.K. (2018). Rejecting or accepting parameter values in Bayesian
  estimation. Advances in Methods and Practices in Psychological Science, 1(2),
  270–280.
Lange et al. (1989). Robust statistical modelling using the t distribution.
  JASA, 84(408), 881–896.
Vehtari et al. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC.
  Statistics and Computing, 27(5), 1413–1432.
"""

import os
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


# Prevents log(0) when WeightedClusterContribution is exactly zero
_OUTCOME_EPS = 1e-9

# Columns written to the user-facing simplified CSV
_SIMPLIFIED_COLS = [
    "Target_Region_ID",
    "Source_Cluster",
    "PosteriorMean",
    "CredibleInterval_2.5",
    "CredibleInterval_97.5",
    "ROPE_Category",
    "StatisticalQuality",
    "BayesianR2",
    "RatioOverlap",
    "WeightedClusterContribution",
    "P90",
]

# Full column order for the raw CSV (all intermediate metrics preserved)
_FULL_COLS = [
    "Target_Region_ID", "Source_Cluster",
    "PosteriorMean", "ROPE_Category", "ROPE_High",
    "StatisticalQuality", "BayesianR2",
    "CredibleInterval_2.5", "CredibleInterval_97.5", "CI_Width", "RelativeUncertainty",
    "ClusterVolume", "OverlapVolume", "SumProb", "P90",
    "RatioOverlap", "DensityOverlap", "WeightedClusterContribution",
]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _write_diagnostic(path, lines):
    """Print *lines* to the console and optionally write them to a file.

    Parameters
    ----------
    path : str or None
        Destination file.  The parent directory must already exist.
        Pass ``None`` to suppress file output (console only).
    lines : list of str
        Lines to emit.
    """
    for line in lines:
        print(line)
    if path is not None:
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")


def _check_image_space(img_a, name_a, img_b, name_b, on_mismatch="error"):
    """Verify that two NIfTI images share the same voxel grid.

    Three properties are checked in order:

    1. **Shape** — spatial dimensions (first three axes) must be identical.
       A shape mismatch always raises ``ValueError`` regardless of
       *on_mismatch*.

    2. **Affine** — checked with ``numpy.allclose`` using an adaptive
       tolerance ``atol = min(1e-3, min_voxel_size)``.

    3. **Orientation codes** — both images are brought to canonical
       orientation via ``nibabel.as_closest_canonical`` and their axis
       codes are compared.

    Parameters
    ----------
    img_a, img_b : nibabel spatial image
    name_a, name_b : str
        Human-readable labels used in error/warning messages.
    on_mismatch : {'error', 'warn'}
        Action on affine or orientation mismatch.  Shape mismatches always
        raise regardless of this setting.

    Returns
    -------
    info : dict
        Keys: ``shape_a``, ``shape_b``, ``affine_a``, ``affine_b``,
        ``atol``, ``orientation_a``, ``orientation_b``, ``issues``.

    Raises
    ------
    ValueError
        On shape mismatch (always) or affine/orientation mismatch when
        *on_mismatch* is ``'error'``.
    """
    shape_a = img_a.shape[:3]
    shape_b = img_b.shape[:3]
    affine_a = img_a.affine
    affine_b = img_b.affine

    min_voxel_size = float(np.min(np.abs(np.diag(affine_a)[:3])))
    atol = min(1.0, min_voxel_size)

    orientation_a = nib.aff2axcodes(nib.as_closest_canonical(img_a).affine)
    orientation_b = nib.aff2axcodes(nib.as_closest_canonical(img_b).affine)

    info = {
        "shape_a": shape_a,
        "shape_b": shape_b,
        "affine_a": affine_a,
        "affine_b": affine_b,
        "atol": atol,
        "orientation_a": orientation_a,
        "orientation_b": orientation_b,
        "issues": [],
    }

    if shape_a != shape_b:
        raise ValueError(
            f"Image space mismatch — SHAPE:\n"
            f"  {name_a}: shape = {shape_a}\n"
            f"  {name_b}: shape = {shape_b}\n"
            f"Both images must have the same voxel grid. "
            f"Check that all inputs are in the same space (e.g. MNI152)."
        )

    if not np.allclose(affine_a, affine_b, atol=atol):
        diff = np.abs(affine_a - affine_b)
        max_diff = float(diff.max())
        issue = (
            f"Affine mismatch between '{name_a}' and '{name_b}' "
            f"(max element-wise difference = {max_diff:.6f}, tolerance = {atol:.6f}).\n"
            f"  {name_a} affine:\n{affine_a}\n"
            f"  {name_b} affine:\n{affine_b}\n"
            f"Tip: ensure both images are registered to the same reference "
            f"and saved with the same voxel size and origin."
        )
        info["issues"].append(issue)

    if orientation_a != orientation_b:
        issue = (
            f"Orientation mismatch between '{name_a}' and '{name_b}' "
            f"after canonicalisation:\n"
            f"  {name_a}: {orientation_a}\n"
            f"  {name_b}: {orientation_b}\n"
            f"Tip: use a consistent NIfTI convention (e.g. convert with "
            f"fslreorient2std or nibabel.as_closest_canonical) before running."
        )
        info["issues"].append(issue)

    if info["issues"]:
        header = (
            f"[WARNING] Image space problem(s) detected comparing "
            f"'{name_a}' and '{name_b}':"
        )
        for issue in info["issues"]:
            if on_mismatch == "error":
                raise ValueError(f"{header}\n{issue}")
            else:
                print(f"{header}")
                print(f"  {issue}")

    return info


def compute_cluster_features(preserved_map, parcellation_path):
    """Compute connectivity features for each parcellation region.

    Parameters
    ----------
    preserved_map : numpy.ndarray
        3D array of preserved connection probabilities:
        cluster_prob × (1 − patient_prob).
    parcellation_path : str
        Path to the parcellation atlas NIfTI image (integer labels).

    Returns
    -------
    df : pandas.DataFrame
        One row per region with ``OverlapVolume > 0``.  Columns:

        - ``Target_Region_ID``: parcellation region integer label
        - ``ClusterVolume``: total voxels in the region
        - ``OverlapVolume``: voxels with preserved connections (value > 0)
        - ``SumProb``: sum of preserved connection values in the region
        - ``P90``: 90th percentile of preserved values in the region
        - ``RatioOverlap``: OverlapVolume / ClusterVolume
        - ``DensityOverlap``: SumProb / OverlapVolume
        - ``WeightedClusterContribution``: SumProb / ClusterVolume

    Raises
    ------
    ValueError
        If *preserved_map* and parcellation shapes do not match.
    """
    parc_img = nib.load(parcellation_path)
    parc_data = parc_img.get_fdata().astype(int)

    if preserved_map.shape != parc_data.shape:
        raise ValueError(
            f"Shape mismatch inside compute_cluster_features: "
            f"preserved map {preserved_map.shape} != "
            f"parcellation {parc_data.shape}. "
            f"Ensure the preserved connection map and the parcellation atlas "
            f"are in the same voxel grid (same shape, same affine)."
        )

    cluster_ids = np.unique(parc_data)
    cluster_ids = cluster_ids[cluster_ids > 0]

    features = []
    for cluster_id in cluster_ids:
        mask = (parc_data == cluster_id)
        cluster_volume = np.sum(mask)
        overlap_mask = mask & (preserved_map > 0)
        overlap_volume = np.sum(overlap_mask)
        sum_prob = np.sum(preserved_map[overlap_mask])
        p90 = np.percentile(preserved_map[overlap_mask], 90) if overlap_volume > 0 else 0
        ratio_overlap = overlap_volume / cluster_volume if cluster_volume > 0 else 0
        density_overlap = sum_prob / overlap_volume if overlap_volume > 0 else 0
        weighted_cluster_contribution = sum_prob / cluster_volume if cluster_volume > 0 else 0
        features.append([cluster_id, cluster_volume, overlap_volume, sum_prob, p90,
                         ratio_overlap, density_overlap, weighted_cluster_contribution])

    df = pd.DataFrame(features,
                      columns=["Target_Region_ID", "ClusterVolume", "OverlapVolume",
                               "SumProb", "P90", "RatioOverlap", "DensityOverlap",
                               "WeightedClusterContribution"])
    df = df[df["OverlapVolume"] > 0]
    return df


def run_bayesian_model(df):
    """Fit a robust Bayesian regression model for preserved connectivity.

    The outcome is ``log(WeightedClusterContribution + ε)`` where
    ``WeightedClusterContribution = SumProb / ClusterVolume``.  Normalising by
    region volume removes the size confound present in raw ``SumProb`` (large
    atlas regions accumulate high sums simply by containing more voxels).

    A Student-t likelihood with estimated degrees-of-freedom ν replaces the
    Normal likelihood to handle residual heteroscedasticity robustly (Lange
    et al., 1989; Gelman et al., 2014, Ch. 17).  The prior
    ``ν ~ Exponential(1/29)`` has mean 29 (close to Normal) while allowing
    heavier tails if the data demands them.  Estimated ν < 5 indicates
    substantial departures from Normality.

    Predictors (standardised to z-scores):
    ``log(ClusterVolume+1)``, ``P90``, ``RatioOverlap``, ``DensityOverlap``.
    ``WeightedClusterContribution`` is the outcome and is therefore excluded
    from the predictor set.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``compute_cluster_features`` with ``Source_Cluster`` column.

    Returns
    -------
    model_results : dict
        Keys:

        - ``trace``: arviz.InferenceData with posterior samples
        - ``bayesian_r2``: float — Bayesian R² (Gelman et al., 2019)
        - ``loo``: LOO-CV results or None
        - ``effect_sizes``: dict of standardised coefficients (effect sizes)
        - ``standardization_params``: dict of predictor means/stds
        - ``nu_mean``: float — posterior mean of ν (degrees of freedom)
        - ``nu_hdi``: tuple — (2.5th, 97.5th) percentile of ν posterior
        - ``outcome_stats``: dict with mean and std of observed outcome y

    References
    ----------
    Lange, K.L., Little, R.J.A., & Taylor, J.M.G. (1989). JASA, 84, 881–896.
    Gelman et al. (2014). Bayesian Data Analysis (3rd ed.). CRC Press.
    Gelman et al. (2019). The American Statistician, 73(3), 307–309.
    Cohen, J. (1988). Statistical Power Analysis (2nd ed.).
    Vehtari et al. (2017). Statistics and Computing, 27(5), 1413–1432.
    """
    # Outcome: log(WeightedClusterContribution + eps)
    y = np.log(df["WeightedClusterContribution"] + _OUTCOME_EPS)

    # Predictors (WCC excluded — it is the outcome)
    log_cluster_vol = np.log(df["ClusterVolume"] + 1)
    X_raw = np.column_stack([
        log_cluster_vol,
        df["P90"].values,
        df["RatioOverlap"].values,
        df["DensityOverlap"].values,
    ])

    # Standardise predictors to z-scores (enables effect-size interpretation)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1
    X = (X_raw - X_mean) / X_std

    predictor_names = [
        "log_ClusterVolume", "P90", "RatioOverlap", "DensityOverlap",
    ]
    standardization_params = {
        name: {"mean": float(m), "std": float(s)}
        for name, m, s in zip(predictor_names, X_mean, X_std)
    }

    with pm.Model() as model:  # noqa: F841
        # Priors for standardised coefficients
        beta0 = pm.Normal("beta0", mu=0, sigma=10)
        beta_cv = pm.Normal("beta_cv", mu=0, sigma=10)
        beta_p90 = pm.Normal("beta_p90", mu=0, sigma=10)
        beta_ro = pm.Normal("beta_ro", mu=0, sigma=10)
        beta_de = pm.Normal("beta_de", mu=0, sigma=10)

        mu = (beta0
              + beta_cv * X[:, 0]
              + beta_p90 * X[:, 1]
              + beta_ro * X[:, 2]
              + beta_de * X[:, 3])
        mu_det = pm.Deterministic("mu_det", mu)

        sigma = pm.HalfNormal("sigma", sigma=5)

        # Robust Student-t likelihood: ν ~ Exp(1/29) has mean 29 (near Normal)
        # but allows heavy tails when residuals are heterogeneous.
        # Reference: Lange et al. (1989); Gelman et al. BDA3 Ch. 17.
        nu = pm.Exponential("nu", lam=1.0 / 29)
        pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=y)

        trace = pm.sample(2000, return_inferencedata=True, random_seed=42,
                          idata_kwargs={"log_likelihood": True})
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # Bayesian R² (Gelman et al., 2019)
    y_pred = trace.posterior_predictive["obs"].values.reshape(-1, len(y))
    var_fitted = np.var(y_pred.mean(axis=0))
    var_residual = np.mean(np.var(y_pred, axis=0))
    bayesian_r2 = var_fitted / (var_fitted + var_residual)

    # LOO-CV (Vehtari et al., 2017)
    try:
        loo = az.loo(trace, pointwise=False)
    except Exception as e:
        print(f"Warning: LOO-CV computation failed: {e}")
        loo = None

    # Standardised effect sizes (posterior means of standardised coefficients)
    effect_sizes = {
        "log_ClusterVolume": float(trace.posterior["beta_cv"].mean()),
        "P90": float(trace.posterior["beta_p90"].mean()),
        "RatioOverlap": float(trace.posterior["beta_ro"].mean()),
        "DensityOverlap": float(trace.posterior["beta_de"].mean()),
    }

    # ν posterior summary
    nu_samples = trace.posterior["nu"].values.flatten()
    nu_mean = float(nu_samples.mean())
    nu_hdi = (float(np.percentile(nu_samples, 2.5)),
              float(np.percentile(nu_samples, 97.5)))

    return {
        "trace": trace,
        "bayesian_r2": float(bayesian_r2),
        "loo": loo,
        "effect_sizes": effect_sizes,
        "standardization_params": standardization_params,
        "nu_mean": nu_mean,
        "nu_hdi": nu_hdi,
        "outcome_stats": {"mean": float(y.mean()), "std": float(y.std())},
    }


def _compute_statistical_quality(posterior_mean, relative_uncertainty,
                                  bayesian_r2, median_all_scores):
    """Assign a statistical quality category to a single observation.

    Categories reflect statistical confidence (model fit × precision × rank),
    not clinical or anatomical importance.

    Parameters
    ----------
    posterior_mean : float
    relative_uncertainty : float
        CI_Width / |PosteriorMean|.
    bayesian_r2 : float
    median_all_scores : float
        Median posterior mean across all observations (computed once per model).

    Returns
    -------
    str

    References
    ----------
    Gelman et al. (2019). R-squared for Bayesian Regression Models.
    Cohen, J. (1988). Statistical Power Analysis (2nd ed.).
    """
    if bayesian_r2 < 0.3:
        return "Poor_Model_Fit"

    high_confidence = relative_uncertainty < 0.5
    high_strength = posterior_mean > median_all_scores

    if bayesian_r2 > 0.7:
        if high_strength and high_confidence:
            return "Strong_High_Confidence"
        elif high_strength:
            return "Strong_Moderate_Confidence"
        elif high_confidence:
            return "Moderate_High_Confidence"
        else:
            return "Moderate_Confidence"
    else:
        if high_confidence:
            return "Weak_Model_High_Confidence"
        else:
            return "Weak_Model_Moderate_Confidence"


def compute_rope_classification(ci_lower, ci_upper, rope_high):
    """Classify a posterior credible interval relative to a ROPE upper bound.

    ROPE (Region of Practical Equivalence) identifies connectivity scores that
    are statistically below a minimum meaningful level — the Bayesian analog of
    "statistically significant but negligible effect size" in frequentist
    statistics.

    The ROPE is one-sided: ``(−∞, rope_high]``.  Pairs with ``ci_upper <=
    rope_high`` have their entire 95 % CI within the negligible zone.  Pairs
    with ``ci_lower > rope_high`` are confidently above it.

    Parameters
    ----------
    ci_lower : float
        Lower bound of the 95 % credible interval for the predicted log-WCC.
    ci_upper : float
        Upper bound of the 95 % credible interval.
    rope_high : float
        Upper boundary of the negligible zone (log-WCC scale).  Default in
        ``generate_report`` is ``mean(y) − 1 × SD(y)`` of the observed outcome
        distribution — approximately the bottom 16 % of a Normal distribution.

    Returns
    -------
    str
        One of:

        - ``"Meaningful"``: ``ci_lower > rope_high`` — connectivity confidently
          above the negligible threshold; candidate target.
        - ``"Negligible"``: ``ci_upper <= rope_high`` — connectivity
          indistinguishable from background; not a viable target.
        - ``"Inconclusive"``: CI straddles the boundary; insufficient data to
          decide.

    References
    ----------
    Kruschke, J.K. (2018). Advances in Methods and Practices in Psychological
      Science, 1(2), 270–280. https://doi.org/10.1177/2515245918771304
    Kruschke, J.K., & Liddell, T.M. (2018). Psychonomic Bulletin & Review,
      25(1), 178–206. https://doi.org/10.3758/s13423-016-1221-4
    """
    if ci_lower > rope_high:
        return "Meaningful"
    elif ci_upper <= rope_high:
        return "Negligible"
    else:
        return "Inconclusive"


def generate_model_summary(model_results, output_path):
    """Write a human-readable model diagnostics file.

    Parameters
    ----------
    model_results : dict
        Output of ``run_bayesian_model``.
    output_path : str
        Destination path for the ``.txt`` summary.

    References
    ----------
    Gelman et al. (2019). R-squared for Bayesian Regression Models.
    Cohen, J. (1988). Statistical Power Analysis (2nd ed.).
    Lange et al. (1989). JASA, 84, 881–896.
    Kruschke, J.K. (2018). Advances in Methods and Practices in Psychological
      Science, 1(2), 270–280.
    Vehtari et al. (2017). Statistics and Computing, 27(5), 1413–1432.
    """
    r2 = model_results["bayesian_r2"]
    effect_sizes = model_results["effect_sizes"]
    loo = model_results["loo"]
    nu_mean = model_results.get("nu_mean")
    nu_hdi = model_results.get("nu_hdi")

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BAYESIAN CONNECTIVITY MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Outcome
        f.write("OUTCOME VARIABLE\n")
        f.write("-" * 40 + "\n")
        f.write("log(WeightedClusterContribution + ε)\n")
        f.write("  where WeightedClusterContribution = SumProb / ClusterVolume\n")
        f.write("  Normalising by region volume removes the size confound\n")
        f.write("  present in raw SumProb (Ashburner & Friston, 2000;\n")
        f.write("  Rorden et al., 2007).\n\n")

        # Likelihood robustness
        f.write("LIKELIHOOD ROBUSTNESS (Student-t)\n")
        f.write("-" * 40 + "\n")
        if nu_mean is not None:
            f.write(f"Estimated ν (degrees of freedom): {nu_mean:.2f}")
            if nu_hdi is not None:
                f.write(f"  [95% HDI: {nu_hdi[0]:.2f}, {nu_hdi[1]:.2f}]")
            f.write("\n")
            if nu_mean < 5:
                interpretation = (
                    "Strong evidence of heavy-tailed residuals. "
                    "Normal likelihood would have been inappropriate."
                )
            elif nu_mean < 30:
                interpretation = (
                    "Moderate departure from Normality. "
                    "Student-t likelihood provides meaningful robustness."
                )
            else:
                interpretation = (
                    "Residuals consistent with Normal distribution. "
                    "Student-t provides robustness without cost."
                )
            f.write(f"Interpretation: {interpretation}\n")
        else:
            f.write("ν not available in model results.\n")
        f.write("(Reference: Lange et al., 1989, JASA 84:881–896;\n")
        f.write(" Gelman et al., 2014, BDA3 Ch. 17)\n\n")

        # Model fit quality
        f.write("MODEL FIT QUALITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Bayesian R²: {r2:.4f}\n\n")
        if r2 > 0.9:
            fit_interp = "Excellent fit"
        elif r2 > 0.7:
            fit_interp = "Good fit"
        elif r2 > 0.5:
            fit_interp = "Moderate fit"
        elif r2 > 0.3:
            fit_interp = "Weak fit"
        else:
            fit_interp = "Poor fit (interpret results with caution)"
        f.write(f"Interpretation: {fit_interp}\n")
        f.write("(R² represents proportion of variance explained;\n")
        f.write(" Gelman et al., 2019)\n\n")

        # Effect sizes
        f.write("EFFECT SIZES (Standardised Coefficients)\n")
        f.write("-" * 40 + "\n")
        f.write("Predictors are z-scored; coefficients are Cohen's d-like\n")
        f.write("effect sizes directly comparable across predictors.\n\n")

        sorted_effects = sorted(effect_sizes.items(),
                                key=lambda x: abs(x[1]), reverse=True)
        for predictor, effect in sorted_effects:
            abs_effect = abs(effect)
            if abs_effect > 0.5:
                magnitude = "Large"
            elif abs_effect > 0.3:
                magnitude = "Medium"
            elif abs_effect > 0.1:
                magnitude = "Small"
            else:
                magnitude = "Negligible"
            direction = "(+)" if effect > 0 else "(-)"
            f.write(f"{predictor:30s}: {effect:7.4f} {direction:3s} [{magnitude}]\n")
        f.write("\n(Cohen 1988 thresholds: |β| > 0.5 = Large, > 0.3 = Medium, > 0.1 = Small)\n\n")

        # LOO-CV
        if loo is not None:
            f.write("PREDICTIVE ACCURACY (LOO-CV)\n")
            f.write("-" * 40 + "\n")
            try:
                f.write(f"LOO: {loo.loo:.2f}\n")
                f.write(f"LOO standard error: {loo.loo_se:.2f}\n")
                f.write(f"p_loo (effective parameters): {loo.p_loo:.2f}\n\n")
                f.write("(Lower LOO indicates better out-of-sample prediction;\n")
                f.write(" Vehtari et al., 2017)\n\n")
            except AttributeError:
                f.write("LOO-CV computed (see full trace for details)\n\n")

        # References
        f.write("\n" + "=" * 60 + "\n")
        f.write("REFERENCES\n")
        f.write("=" * 60 + "\n")
        f.write("Ashburner, J., & Friston, K.J. (2000). Voxel-based morphometry\n")
        f.write("  — the methods. NeuroImage, 11(6), 805–821.\n\n")
        f.write("Cohen, J. (1988). Statistical Power Analysis for the Behavioural\n")
        f.write("  Sciences (2nd ed.). Lawrence Erlbaum Associates.\n\n")
        f.write("Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A.,\n")
        f.write("  & Rubin, D.B. (2014). Bayesian Data Analysis (3rd ed.). CRC Press.\n\n")
        f.write("Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019).\n")
        f.write("  R-squared for Bayesian Regression Models. The American\n")
        f.write("  Statistician, 73(3), 307–309.\n\n")
        f.write("Kruschke, J.K. (2018). Rejecting or accepting parameter values in\n")
        f.write("  Bayesian estimation. Advances in Methods and Practices in\n")
        f.write("  Psychological Science, 1(2), 270–280.\n\n")
        f.write("Lange, K.L., Little, R.J.A., & Taylor, J.M.G. (1989). Robust\n")
        f.write("  statistical modelling using the t distribution. JASA, 84, 881–896.\n\n")
        f.write("Rorden, C., Karnath, H.-O., & Bonilha, L. (2007). Improving\n")
        f.write("  lesion-symptom mapping. J. Cognitive Neuroscience, 19(7), 1081–1088.\n\n")
        f.write("Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian\n")
        f.write("  model evaluation using LOO-CV and WAIC.\n")
        f.write("  Statistics and Computing, 27(5), 1413–1432.\n")

    print(f"Model summary saved: {output_path}")


def generate_report(df, model_results, output_csv, rope_high=None):
    """Build ranked CSV reports and a connectivity-score plot.

    Writes two CSV files and one PNG:

    - ``<output_csv>``: simplified user-facing CSV with key columns only.
    - ``<output_csv_base>_full.csv``: raw CSV with all intermediate metrics.
    - ``<output_csv_base>_plot.png``: error-bar plot of posterior scores.

    The simplified CSV is intended for direct clinical/scientific interpretation.
    The full CSV preserves all intermediate quantities for re-analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature DataFrame from ``compute_cluster_features`` (with
        ``Source_Cluster`` column).
    model_results : dict
        Output of ``run_bayesian_model``.
    output_csv : str
        Path for the simplified CSV.  The full CSV is derived by inserting
        ``_full`` before ``.csv``.
    rope_high : float or None, optional
        Upper bound of the negligible zone on the log-WCC scale.  When
        ``None``, defaults to the 25th percentile of the raw observed
        log(WCC) distribution (pre-model, no dependence on posterior).
        Pairs whose entire 95 % CI lies below this threshold are labelled
        ``"Negligible"``.

    Notes
    -----
    Statistical quality categories reflect model fit and estimate precision,
    not clinical importance.  The ``ROPE_Category`` column provides a
    complementary interpretation: ``"Meaningful"`` pairs have their entire 95 %
    CI above the negligible threshold; ``"Negligible"`` pairs are
    indistinguishable from background connectivity.

    References
    ----------
    Kruschke, J.K. (2018). Advances in Methods and Practices in Psychological
      Science, 1(2), 270–280.
    """
    if df.empty:
        print("No preserved connections were found. Skipping report generation.")
        return

    df = df.copy()  # avoid mutating the caller's dataframe

    trace = model_results["trace"]
    bayesian_r2 = model_results["bayesian_r2"]

    mu_samples = trace.posterior["mu_det"]
    combined_samples = mu_samples.stack(sample=("chain", "draw")).values
    posterior_mean = combined_samples.mean(axis=1)
    ci_lower = np.percentile(combined_samples, 2.5, axis=1)
    ci_upper = np.percentile(combined_samples, 97.5, axis=1)
    ci_width = ci_upper - ci_lower

    # Guard against near-zero posterior mean in RelativeUncertainty
    relative_uncertainty = np.where(
        np.abs(posterior_mean) < 1e-9,
        np.inf,
        ci_width / np.abs(posterior_mean),
    )

    df["PosteriorMean"] = posterior_mean
    df["CredibleInterval_2.5"] = ci_lower
    df["CredibleInterval_97.5"] = ci_upper
    df["CI_Width"] = ci_width
    df["RelativeUncertainty"] = relative_uncertainty
    df["BayesianR2"] = bayesian_r2

    # ROPE classification
    # Default threshold: 25th percentile of the RAW observed log(WCC).
    # Using the raw (pre-model) distribution avoids any circularity with the
    # posterior: the threshold is fixed before sampling and does not depend
    # on model output.  mean(raw_y) - 1*SD(raw_y) was the original intent but
    # placed the boundary far below the floor of model predictions (because
    # regression smooths extreme raw values toward the mean).  The 25th
    # percentile of the raw distribution is more robust: it places the boundary
    # at the value below which the bottom quarter of *observed* connectivity
    # scores lie, which is a pre-specified, data-quality criterion independent
    # of the posterior.
    if rope_high is None:
        raw_log_wcc = np.log(df["WeightedClusterContribution"].values + _OUTCOME_EPS)
        rope_high = float(np.percentile(raw_log_wcc, 25))
    df["ROPE_High"] = rope_high
    df["ROPE_Category"] = [
        compute_rope_classification(lo, hi, rope_high)
        for lo, hi in zip(ci_lower, ci_upper)
    ]

    median_all_scores = float(np.median(posterior_mean))
    df["StatisticalQuality"] = [
        _compute_statistical_quality(pm, ru, bayesian_r2, median_all_scores)
        for pm, ru in zip(posterior_mean, relative_uncertainty)
    ]

    df = df.sort_values(by="PosteriorMean", ascending=False)

    # --- Full CSV (all metrics) ---
    full_cols = [c for c in _FULL_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in _FULL_COLS]
    full_csv = output_csv.replace("_analysis_report.csv", "_analysis_report_full.csv")
    df[full_cols + extra].to_csv(full_csv, index=False)

    # --- Simplified CSV (user-facing) ---
    simp_cols = [c for c in _SIMPLIFIED_COLS if c in df.columns]
    df[simp_cols].to_csv(output_csv, index=False)

    # Generate model summary
    summary_path = output_csv.replace(".csv", "_model_summary.txt")
    generate_model_summary(model_results, summary_path)

    # Plot: posterior means with 95 % credible intervals
    labels = [
        f"R{row['Target_Region_ID']}:{row['Source_Cluster'][:10]}"
        for _, row in df.iterrows()
    ]
    plt.figure(figsize=(max(10, len(df) * 0.3), 6))
    plt.errorbar(
        x=np.arange(len(df)),
        y=df["PosteriorMean"],
        yerr=[df["PosteriorMean"] - df["CredibleInterval_2.5"],
              df["CredibleInterval_97.5"] - df["PosteriorMean"]],
        fmt="o", capsize=5,
    )
    plt.xticks(np.arange(len(df)), labels, rotation=90)
    plt.xlabel("Target Region : Source Cluster")
    plt.ylabel("log(WeightedClusterContribution) — Posterior Estimate")
    plt.title(f"Posterior Connectivity Scores (Bayesian R²={bayesian_r2:.3f})")
    plt.tight_layout()
    plot_filename = output_csv.replace(".csv", "_plot.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    print(f"Simplified report: {output_csv}")
    print(f"Full report:       {full_csv}")
    print(f"Plot saved:        {plot_filename}")


def translate_cluster_labels(atlas_txt_path, input_csv, output_csv):
    """Add anatomical region names to a report CSV and write a translated copy.

    Reads the atlas label file (tab-separated, with ``color`` and ``nom_c``
    columns), maps ``Target_Region_ID`` values to names, inserts
    ``Target_Region_Name`` immediately after ``Target_Region_ID``, and writes
    the result to *output_csv*.  Works with both the simplified and full CSV
    output from ``generate_report``.

    Parameters
    ----------
    atlas_txt_path : str
        Path to the atlas TSV file with ``color`` (integer ID) and
        ``nom_c`` (anatomical name) columns.
    input_csv : str
        Path to an input report CSV containing ``Target_Region_ID``.
    output_csv : str
        Path to write the translated report.
    """
    atlas_df = pd.read_csv(atlas_txt_path, sep="\t")
    result_df = pd.read_csv(input_csv)

    valid_regions = set(result_df["Target_Region_ID"])
    label_mapping = {
        color: nom_c
        for color, nom_c in zip(atlas_df["color"], atlas_df["nom_c"])
        if color in valid_regions
    }

    result_df.insert(
        result_df.columns.get_loc("Target_Region_ID") + 1,
        "Target_Region_Name",
        result_df["Target_Region_ID"].map(label_mapping),
    )

    result_df.to_csv(output_csv, index=False)
    print(f"Translated report saved: {output_csv}")


def run_connectivity_analysis(
    patient_disconnectome_paths,
    cluster_disconnectome_paths,
    parcellation_path,
    output_folder,
    atlas_txt_path=None,
    on_space_mismatch="error",
    min_ratio_overlap=None,
    min_overlap_voxels=None,
    rope_high=None,
):
    """Run the complete preserved-connectivity analysis pipeline.

    For each patient × cluster combination, computes:
    ``preserved = cluster_prob × (1 − patient_prob)``

    All cluster-region combinations for a patient are analysed in a single
    Bayesian model so that posterior scores are directly comparable across
    clusters.

    Parameters
    ----------
    patient_disconnectome_paths : list or str
        Path(s) to patient disconnectome NIfTI images (probability maps, 0–1).
    cluster_disconnectome_paths : list or str
        Path(s) to cluster disconnectome NIfTI images.
    parcellation_path : str
        Path to the parcellation atlas NIfTI image (integer labels).
    output_folder : str
        Folder where all output files are written.
    atlas_txt_path : str, optional
        Atlas TSV for translating region IDs to anatomical names.
    on_space_mismatch : {'error', 'warn'}, optional
        Action on affine/orientation mismatch.  Shape mismatches always abort.
    min_ratio_overlap : float or None, optional
        **Detection-limit filter (size-normalised).**  Exclude region-cluster
        pairs where ``RatioOverlap < min_ratio_overlap`` before fitting the
        model.  Because ``RatioOverlap = OverlapVolume / ClusterVolume``, this
        threshold scales with region size, avoiding the bias in a fixed-voxel
        threshold.

        .. warning::
           Set this value based on anatomical/spatial-resolution grounds
           (e.g. "at least 2 % of the region must overlap"), **not** to
           improve statistical results after inspecting the output.
           Post-hoc threshold tuning is equivalent to p-hacking.

    min_overlap_voxels : int or None, optional
        **Absolute detection floor.**  Exclude pairs with fewer than this many
        overlap voxels regardless of region size.  Guards against single-voxel
        partial-volume artefacts.  Applied alongside *min_ratio_overlap* with
        AND logic.

        .. warning:: Same caution as *min_ratio_overlap* applies.

    rope_high : float or None, optional
        Upper bound of the negligible zone on log-WCC scale for ROPE
        classification (see ``compute_rope_classification``).  When ``None``,
        computed as the 25th percentile of the raw observed log(WCC)
        distribution before model fitting.

    Output files per patient
    ------------------------
    - ``<patient>_analysis_report.csv``: simplified user-facing CSV.
    - ``<patient>_analysis_report_full.csv``: full CSV with all metrics.
    - ``<patient>_analysis_report_model_summary.txt``: model diagnostics.
    - ``<patient>_analysis_report_plot.png``: connectivity-score plot.
    - ``translated_<patient>_analysis_report.csv``: (if atlas provided)
    - ``translated_<patient>_analysis_report_full.csv``: (if atlas provided)
    - ``derivatives/<patient>/<patient>_<cluster>_preserved.nii.gz``.
    - ``<patient>_FAILED_diagnostic.txt``: written when no overlaps found.
    - ``command_used.txt``: the command line used.
    """
    os.makedirs(output_folder, exist_ok=True)

    command_used = " ".join(sys.argv)
    with open(os.path.join(output_folder, "command_used.txt"), "w") as f:
        f.write(command_used + "\n")

    if isinstance(patient_disconnectome_paths, str):
        patient_disconnectome_paths = [patient_disconnectome_paths]
    if isinstance(cluster_disconnectome_paths, str):
        cluster_disconnectome_paths = [cluster_disconnectome_paths]

    parc_img = nib.load(parcellation_path)
    parc_n_regions = len(np.unique(parc_img.get_fdata().astype(int))) - 1
    parc_shape = parc_img.shape[:3]

    for patient_path in patient_disconnectome_paths:
        patient_basename = Path(patient_path).stem
        print(f"Processing patient: {patient_basename}")

        patient_img = nib.load(patient_path)
        patient_data = patient_img.get_fdata()

        _check_image_space(
            patient_img, patient_basename,
            parc_img, Path(parcellation_path).name,
            on_mismatch=on_space_mismatch,
        )

        cluster_diagnostics = []
        all_dfs = []

        for cluster_path in cluster_disconnectome_paths:
            cluster_name = Path(cluster_path).stem
            print(f"  Processing cluster: {cluster_name}")

            cluster_img = nib.load(cluster_path)
            _check_image_space(
                patient_img, patient_basename,
                cluster_img, cluster_name,
                on_mismatch=on_space_mismatch,
            )
            cluster_data = cluster_img.get_fdata()

            try:
                preserved = cluster_data * (1 - patient_data)
            except ValueError as exc:
                raise ValueError(
                    f"Could not compute preserved connections for patient "
                    f"'{patient_basename}' and cluster '{cluster_name}'.\n"
                    f"Patient shape: {patient_data.shape}, "
                    f"Cluster shape: {cluster_data.shape}.\n"
                    f"Original numpy error: {exc}"
                ) from exc

            n_nonzero_preserved = int(np.sum(preserved > 0))

            derivatives_folder = os.path.join(
                output_folder, "derivatives", patient_basename
            )
            os.makedirs(derivatives_folder, exist_ok=True)
            preserved_img = nib.Nifti1Image(preserved, affine=cluster_img.affine)
            preserved_path = os.path.join(
                derivatives_folder,
                f"{patient_basename}_{cluster_name}_preserved.nii.gz",
            )
            nib.save(preserved_img, preserved_path)
            print(f"    Saved preserved image: {preserved_path}")

            df = compute_cluster_features(preserved, parcellation_path)
            df["Source_Cluster"] = cluster_name
            n_matched_regions = len(df)

            cluster_diagnostics.append({
                "cluster": cluster_name,
                "cluster_shape": cluster_data.shape,
                "n_nonzero_preserved": n_nonzero_preserved,
                "n_matched_regions": n_matched_regions,
            })

            if df.empty:
                print(
                    f"  [WARNING] Cluster '{cluster_name}': the preserved connection map "
                    f"has {n_nonzero_preserved} non-zero voxel(s) but none overlapped with "
                    f"any parcellation region. This cluster contributes 0 observations to "
                    f"the model."
                )

            all_dfs.append(df)

        combined_df = (
            pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        )

        # --- Detection-limit filters (T7) ---
        if not combined_df.empty and (
            min_ratio_overlap is not None or min_overlap_voxels is not None
        ):
            mask = pd.Series(True, index=combined_df.index)
            if min_ratio_overlap is not None:
                mask &= combined_df["RatioOverlap"] >= min_ratio_overlap
            if min_overlap_voxels is not None:
                mask &= combined_df["OverlapVolume"] >= min_overlap_voxels
            n_removed = int((~mask).sum())
            if n_removed > 0:
                pct = 100.0 * n_removed / len(combined_df)
                print(
                    f"  Detection-limit filter removed {n_removed} pairs "
                    f"({pct:.1f}%)"
                )
                if pct > 20:
                    print(
                        "  [WARNING] > 20 % of pairs removed by detection-limit "
                        "filter. Verify thresholds are based on spatial resolution "
                        "grounds, not on inspecting statistical results."
                    )
            combined_df = combined_df[mask].reset_index(drop=True)

        print(f"  Total cluster-region combinations after filtering: {len(combined_df)}")

        if combined_df.empty:
            diag_path = os.path.join(
                output_folder, f"{patient_basename}_FAILED_diagnostic.txt"
            )
            diag_lines = [
                "=" * 70,
                "ANALYSIS FAILED — no cluster-region overlaps found",
                f"Patient: {patient_basename}  ({patient_path})",
                "=" * 70,
                "",
                "INPUT SUMMARY",
                "-" * 40,
                f"  Patient disconnectome : {patient_path}",
                f"    shape              : {patient_data.shape}",
                f"    n non-zero voxels  : {int(np.sum(patient_data > 0))}",
                f"  Parcellation         : {parcellation_path}",
                f"    shape              : {parc_shape}",
                f"    n regions (labels) : {parc_n_regions}",
                "",
                "PER-CLUSTER BREAKDOWN",
                "-" * 40,
            ]
            for cd in cluster_diagnostics:
                diag_lines += [
                    f"  Cluster : {cd['cluster']}",
                    f"    shape                   : {cd['cluster_shape']}",
                    f"    non-zero voxels in map  : {cd['n_nonzero_preserved']}",
                    f"    parcellation regions hit: {cd['n_matched_regions']}",
                    "",
                ]
            diag_lines += [
                "POSSIBLE CAUSES",
                "-" * 40,
                "  1. The preserved connection map is all zeros.",
                "     → The patient disconnectome may cover the entire cluster region.",
                "       Check that patient_prob values are in [0, 1].",
                "  2. The preserved map and parcellation do not spatially overlap.",
                "     → Confirm both are in the same reference space (e.g. MNI152).",
                "     → A hemisphere-specific parcellation will show 0 overlap if the",
                "       preserved connections are entirely in the other hemisphere.",
                "  3. Image affine or voxel-size mismatch (silent misregistration).",
                "     → Re-run with -sm warn to see space-check details.",
                "  4. All cluster voxel values are zero (empty cluster map).",
                "  5. Detection-limit filter removed all remaining pairs.",
                "     → If --min_ratio_overlap or --min_overlap_voxels are set,",
                "       verify the thresholds are appropriate for this data.",
                "",
                "No output report was generated for this patient.",
                "=" * 70,
            ]
            _write_diagnostic(diag_path, diag_lines)
            print(
                f"  [ERROR] Skipping patient '{patient_basename}': "
                f"no cluster-region overlaps found. "
                f"Diagnostic written to: {diag_path}"
            )
            continue

        model_results = run_bayesian_model(combined_df)

        report_csv = os.path.join(
            output_folder, f"{patient_basename}_analysis_report.csv"
        )
        generate_report(combined_df, model_results, report_csv, rope_high=rope_high)

        if atlas_txt_path:
            full_csv = report_csv.replace(
                "_analysis_report.csv", "_analysis_report_full.csv"
            )
            translated_simp = os.path.join(
                output_folder,
                f"translated_{patient_basename}_analysis_report.csv",
            )
            translated_full = os.path.join(
                output_folder,
                f"translated_{patient_basename}_analysis_report_full.csv",
            )
            translate_cluster_labels(atlas_txt_path, report_csv, translated_simp)
            translate_cluster_labels(atlas_txt_path, full_csv, translated_full)

        print(f"Finished processing patient: {patient_basename}")
