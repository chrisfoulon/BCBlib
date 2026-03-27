"""
Module: best_overlap (formerly connectivity)
Description: Provides functions to process disconnectome maps by computing preserved connections
             using probability multiplication (cluster disconnection probability × probability of NOT
             being disconnected by patient), computing connectivity features for parcellation atlas
             regions, running a Bayesian model to obtain latent connectivity scores, and generating
             CSV reports (including uncertainty measures). Supports analyzing multiple cluster
             disconnectomes simultaneously in a single model for direct comparison of connectivity
             strength across clusters and regions. Optionally, region IDs can be translated to
             anatomical names using a label file.
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


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _write_diagnostic(path, lines):
    """Print *lines* to the console and, when *path* is not None, also write
    them to a plain-text file.  Each element of *lines* is a string; a
    newline is appended automatically.

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
       *on_mismatch*, because the downstream computation would be
       numerically meaningless (numpy would either broadcast incorrectly or
       crash with a cryptic message).

    2. **Affine** — checked with ``numpy.allclose`` using an adaptive
       tolerance ``atol = min(1e-3, min_voxel_size)`` so that the tolerance
       is always smaller than one voxel (sub-voxel precision) *and* never
       looser than 1 mm (sub-millimetre precision for large-voxel data).

    3. **Orientation codes** — both images are brought to canonical
       orientation via ``nibabel.as_closest_canonical`` and their axis
       codes (e.g. ``('R', 'A', 'S')``) are compared.  A mismatch after
       canonicalisation indicates a genuine orientation difference, not
       merely a storage convention difference.

    Parameters
    ----------
    img_a, img_b : nibabel spatial image
        Loaded NIfTI images to compare.
    name_a, name_b : str
        Human-readable labels for the images (used in error/warning messages).
    on_mismatch : {'error', 'warn'}
        What to do when an affine or orientation mismatch is detected.
        ``'error'`` (default) raises ``ValueError``; ``'warn'`` prints a
        warning and returns the diagnostic information so the caller can
        decide whether to proceed.

    Returns
    -------
    info : dict
        Diagnostic information with keys:
        ``shape_a``, ``shape_b``, ``affine_a``, ``affine_b``,
        ``atol``, ``orientation_a``, ``orientation_b``,
        ``issues`` (list of human-readable issue strings, empty when all
        checks pass).

    Raises
    ------
    ValueError
        On shape mismatch (always) or on affine/orientation mismatch when
        *on_mismatch* is ``'error'``.
    """
    shape_a = img_a.shape[:3]
    shape_b = img_b.shape[:3]
    affine_a = img_a.affine
    affine_b = img_b.affine

    # Adaptive tolerance: whichever is stricter — 1 mm (submillimetre cap) or
    # one voxel size (sub-voxel precision when voxels are smaller than 1 mm).
    # For typical medical imaging this gives 1.0 mm for large voxels and
    # matches the voxel size for high-resolution sub-millimetre data.
    min_voxel_size = float(np.min(np.abs(np.diag(affine_a)[:3])))
    atol = min(1.0, min_voxel_size)

    # Orientation codes after canonicalisation
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

    # --- 1. Shape check (always fatal) ---
    if shape_a != shape_b:
        raise ValueError(
            f"Image space mismatch — SHAPE:\n"
            f"  {name_a}: shape = {shape_a}\n"
            f"  {name_b}: shape = {shape_b}\n"
            f"Both images must have the same voxel grid. "
            f"Check that all inputs are in the same space (e.g. MNI152)."
        )

    # --- 2. Affine check ---
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

    # --- 3. Orientation check (post-canonicalisation) ---
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

    # --- Handle detected issues ---
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
    """
    Compute connectivity features for each parcellation region using a preserved connection map
    and a parcellation atlas.

    Parameters
    ----------
    preserved_map : numpy.ndarray
        A 3D array representing the preserved connections, computed as the probability
        that a connection is affected in the cluster but NOT disconnected by the patient:
        cluster_prob × (1 - patient_prob).
    parcellation_path : str
        File path to the parcellation atlas (NIfTI image) where voxels are labeled by cluster number.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing computed features for each parcellation region. Columns include:
          - Target_Region_ID: Parcellation region ID (integer)
          - ClusterVolume: Total number of voxels in the region.
          - OverlapVolume: Number of voxels in the region with preserved connections (value > 0).
          - SumProb: Sum of preserved connection values within the region.
          - P90: 90th percentile of the preserved connection values in the region.
          - RatioOverlap: OverlapVolume divided by ClusterVolume.
          - DensityOverlap: SumProb divided by OverlapVolume.
          - WeightedClusterContribution: SumProb divided by ClusterVolume.
    """
    parc_img = nib.load(parcellation_path)
    parc_data = parc_img.get_fdata().astype(int)

    # Belt-and-suspenders: catch shape mismatch here even if the caller
    # already called _check_image_space (compute_cluster_features is public).
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
                      columns=["Target_Region_ID", "ClusterVolume", "OverlapVolume", "SumProb", "P90",
                               "RatioOverlap", "DensityOverlap", "WeightedClusterContribution"])
    df = df[df["OverlapVolume"] > 0]
    return df

def run_bayesian_model(df):
    """
    Build and sample from a Bayesian regression model to predict a composite connectivity measure.
    Predictors are standardized to enable interpretation of coefficients as effect sizes.
    Computes Bayesian R² (Gelman et al. 2019) and LOO-CV for model evaluation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cluster features (from compute_cluster_features).

    Returns
    -------
    model_results : dict
        Dictionary containing:
        - 'trace': arviz.InferenceData with posterior samples
        - 'bayesian_r2': Bayesian R² value (model fit quality)
        - 'loo': LOO-CV results (predictive accuracy)
        - 'effect_sizes': dict of standardized coefficients (effect sizes)
        - 'standardization_params': dict of means and stds used for standardization

    Model Details
    -------------
    - Outcome: log(SumProb+1), reflecting both the extent and intensity of preserved connections.
    - Predictors: Standardized versions of log(ClusterVolume+1), P90, RatioOverlap, 
                  DensityOverlap, and WeightedClusterContribution.
    - Standardized coefficients can be interpreted as effect sizes (Cohen 1988).
    - Bayesian R² measures proportion of variance explained (Gelman et al. 2019).
    - LOO-CV assesses out-of-sample predictive accuracy (Vehtari et al. 2017).
    """
    # Prepare outcome variable
    y = np.log(df["SumProb"] + 1)
    
    # Prepare and standardize predictors
    log_cluster_vol = np.log(df["ClusterVolume"] + 1)
    
    # Collect predictors
    X_raw = np.column_stack([
        log_cluster_vol,
        df["P90"].values,
        df["RatioOverlap"].values,
        df["DensityOverlap"].values,
        df["WeightedClusterContribution"].values
    ])
    
    # Standardize (z-score transformation)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X = (X_raw - X_mean) / X_std
    
    # Store standardization parameters for later use
    predictor_names = ["log_ClusterVolume", "P90", "RatioOverlap", 
                      "DensityOverlap", "WeightedClusterContribution"]
    standardization_params = {
        name: {"mean": float(m), "std": float(s)} 
        for name, m, s in zip(predictor_names, X_mean, X_std)
    }
    
    # Build Bayesian model with standardized predictors
    with pm.Model() as model:
        # Priors for standardized coefficients (these are effect sizes)
        beta0    = pm.Normal("beta0", mu=0, sigma=10)
        beta_cv  = pm.Normal("beta_cv", mu=0, sigma=10)
        beta_p90 = pm.Normal("beta_p90", mu=0, sigma=10)
        beta_ro  = pm.Normal("beta_ro", mu=0, sigma=10)
        beta_de  = pm.Normal("beta_de", mu=0, sigma=10)
        beta_wc  = pm.Normal("beta_wc", mu=0, sigma=10)

        # Linear predictor
        mu = (beta0 + beta_cv * X[:, 0] + beta_p90 * X[:, 1] +
              beta_ro * X[:, 2] + beta_de * X[:, 3] + beta_wc * X[:, 4])
        mu_det = pm.Deterministic("mu_det", mu)

        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        # Sample from posterior with log_likelihood for LOO-CV
        trace = pm.sample(2000, return_inferencedata=True, random_seed=42,
                         idata_kwargs={"log_likelihood": True})
        
        # Compute posterior predictive for R² calculation
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # Compute Bayesian R² (Gelman et al. 2019)
    # R² = Var(fitted) / (Var(fitted) + Var(residual))
    y_pred = trace.posterior_predictive["obs"].values.reshape(-1, len(y))
    var_fitted = np.var(y_pred.mean(axis=0))
    var_residual = np.mean(np.var(y_pred, axis=0))
    bayesian_r2 = var_fitted / (var_fitted + var_residual)
    
    # Compute LOO-CV (Vehtari et al. 2017)
    try:
        loo = az.loo(trace, pointwise=False)
    except Exception as e:
        print(f"Warning: LOO-CV computation failed: {e}")
        loo = None
    
    # Extract standardized effect sizes (posterior means of standardized coefficients)
    effect_sizes = {
        "log_ClusterVolume": float(trace.posterior["beta_cv"].mean()),
        "P90": float(trace.posterior["beta_p90"].mean()),
        "RatioOverlap": float(trace.posterior["beta_ro"].mean()),
        "DensityOverlap": float(trace.posterior["beta_de"].mean()),
        "WeightedClusterContribution": float(trace.posterior["beta_wc"].mean())
    }
    
    return {
        'trace': trace,
        'bayesian_r2': float(bayesian_r2),
        'loo': loo,
        'effect_sizes': effect_sizes,
        'standardization_params': standardization_params
    }


def _compute_statistical_quality(posterior_mean, relative_uncertainty, bayesian_r2, median_all_scores):
    """
    Compute statistical quality categories based on model fit, strength of association,
    and confidence in estimates. Categories reflect statistical confidence, not clinical
    importance.
    
    Parameters
    ----------
    posterior_mean : float
        Posterior mean of connectivity score
    relative_uncertainty : float
        Relative width of credible interval (CI_width / posterior_mean)
    bayesian_r2 : float
        Bayesian R² (model fit quality)
    median_all_scores : float
        Median of all posterior means (computed once for efficiency)
    
    Returns
    -------
    str
        Statistical quality category
        
    References
    ----------
    - Gelman et al. (2019). R-squared for Bayesian Regression Models. 
      The American Statistician, 73(3), 307-309.
    - Cohen (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    # Gate on model quality
    if bayesian_r2 < 0.3:
        return "Poor_Model_Fit"
    
    # Determine if uncertainty is acceptable (CI width < 50% of estimate)
    # This is a conservative threshold from uncertainty quantification literature
    high_confidence = relative_uncertainty < 0.5
    
    # Strength based on whether above median (could use absolute threshold if established)
    # For now using relative to allow interpretation even with varying scales
    high_strength = posterior_mean > median_all_scores
    
    # Combine criteria following hierarchical decision tree
    if bayesian_r2 > 0.7:  # Good model fit
        if high_strength and high_confidence:
            return "Strong_High_Confidence"
        elif high_strength:
            return "Strong_Moderate_Confidence"
        elif high_confidence:
            return "Moderate_High_Confidence"
        else:
            return "Moderate_Confidence"
    else:  # 0.3 <= R² <= 0.7: Weak to moderate model fit
        if high_confidence:
            return "Weak_Model_High_Confidence"
        else:
            return "Weak_Model_Moderate_Confidence"


def generate_model_summary(model_results, output_path):
    """
    Generate a text summary file with model diagnostics and effect sizes.
    
    Parameters
    ----------
    model_results : dict
        Dictionary from run_bayesian_model containing:
        - bayesian_r2: Model fit quality
        - effect_sizes: Standardized coefficients
        - loo: LOO-CV results
    output_path : str
        Path to save the summary file
        
    References
    ----------
    - Gelman et al. (2019). R-squared for Bayesian Regression Models.
    - Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
    - Vehtari et al. (2017). Practical Bayesian model evaluation using leave-one-out 
      cross-validation and WAIC. Statistics and Computing, 27(5), 1413-1432.
    """
    r2 = model_results['bayesian_r2']
    effect_sizes = model_results['effect_sizes']
    loo = model_results['loo']
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\\n")
        f.write("BAYESIAN CONNECTIVITY MODEL SUMMARY\\n")
        f.write("="*60 + "\\n\\n")
        
        # Model Fit Quality
        f.write("MODEL FIT QUALITY\\n")
        f.write("-" * 40 + "\\n")
        f.write(f"Bayesian R²: {r2:.4f}\\n\\n")
        
        if r2 > 0.9:
            interpretation = "Excellent fit"
        elif r2 > 0.7:
            interpretation = "Good fit"
        elif r2 > 0.5:
            interpretation = "Moderate fit"
        elif r2 > 0.3:
            interpretation = "Weak fit"
        else:
            interpretation = "Poor fit (interpret results with caution)"
        f.write(f"Interpretation: {interpretation}\\n")
        f.write(f"(R² represents proportion of variance explained)\\n\\n")
        
        # Effect Sizes (Standardized Coefficients)
        f.write("EFFECT SIZES (Standardized Coefficients)\\n")
        f.write("-" * 40 + "\\n")
        f.write("Effect sizes indicate the strength of association between\\n")
        f.write("each predictor and connectivity score.\\n\\n")
        
        # Sort by absolute effect size
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
            f.write(f"{predictor:30s}: {effect:7.4f} {direction:3s} [{magnitude}]\\n")
        
        f.write("\\n(Cohen 1988 thresholds: |β| > 0.5 = Large, > 0.3 = Medium, > 0.1 = Small)\\n\\n")
        
        # LOO-CV
        if loo is not None:
            f.write("PREDICTIVE ACCURACY (LOO-CV)\\n")
            f.write("-" * 40 + "\\n")
            try:
                f.write(f"LOO: {loo.loo:.2f}\\n")
                f.write(f"LOO standard error: {loo.loo_se:.2f}\\n")
                f.write(f"p_loo (effective parameters): {loo.p_loo:.2f}\\n\\n")
                f.write("(Lower LOO indicates better out-of-sample prediction)\\n\\n")
            except AttributeError:
                f.write(f"LOO-CV computed (see full trace for details)\\n\\n")
        
        # References
        f.write("\\n" + "="*60 + "\\n")
        f.write("REFERENCES\\n")
        f.write("="*60 + "\\n")
        f.write("Cohen, J. (1988). Statistical Power Analysis for the Behavioral\\n")
        f.write("  Sciences (2nd ed.). Lawrence Erlbaum Associates.\\n\\n")
        f.write("Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019).\\n")
        f.write("  R-squared for Bayesian Regression Models. The American\\n")
        f.write("  Statistician, 73(3), 307-309.\\n\\n")
        f.write("Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian\\n")
        f.write("  model evaluation using leave-one-out cross-validation and WAIC.\\n")
        f.write("  Statistics and Computing, 27(5), 1413-1432.\\n")
    
    print(f"Model summary saved: {output_path}")


def generate_report(df, model_results, output_csv):
    """
    Extract posterior summaries for the latent connectivity score, rank all cluster-region
    combinations, and generate a CSV report that includes uncertainty measures, statistical
    quality categories, and model fit metrics. The plot and model summary are saved to disk.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with cluster features
    model_results : dict
        Dictionary from run_bayesian_model containing trace, R², LOO, effect sizes
    output_csv : str
        Path to save the CSV report
        
    Notes
    -----
    Statistical quality categories reflect model fit and confidence in estimates,
    not clinical or anatomical importance. Raw metrics (P90, RatioOverlap, etc.)
    must be interpreted in domain context.
    """
    # Check if any preserved connections were found
    if df.empty:
        print("No preserved connections were found. Skipping report generation.")
        return

    trace = model_results['trace']
    bayesian_r2 = model_results['bayesian_r2']
    
    mu_samples = trace.posterior["mu_det"]
    combined_samples = mu_samples.stack(sample=("chain", "draw")).values
    posterior_mean = combined_samples.mean(axis=1)
    ci_lower = np.percentile(combined_samples, 2.5, axis=1)
    ci_upper = np.percentile(combined_samples, 97.5, axis=1)
    ci_width = ci_upper - ci_lower
    relative_uncertainty = ci_width / np.abs(posterior_mean)

    df["PosteriorMean"] = posterior_mean
    df["CredibleInterval_2.5"] = ci_lower
    df["CredibleInterval_97.5"] = ci_upper
    df["CI_Width"] = ci_width
    df["RelativeUncertainty"] = relative_uncertainty
    df["BayesianR2"] = bayesian_r2

    # Compute median of all posterior means once for efficiency
    median_all_scores = np.median(posterior_mean)
    
    # Compute statistical quality categories
    statistical_quality = [
        _compute_statistical_quality(pm, ru, bayesian_r2, median_all_scores)
        for pm, ru in zip(posterior_mean, relative_uncertainty)
    ]
    df["StatisticalQuality"] = statistical_quality

    # Sort by posterior mean (descending) for global ranking
    df = df.sort_values(by="PosteriorMean", ascending=False)
    
    # Reorder columns to put identifiers first, then statistical metrics, then raw features
    column_order = ["Target_Region_ID", "Source_Cluster", 
                    "PosteriorMean", "StatisticalQuality", "BayesianR2",
                    "CredibleInterval_2.5", "CredibleInterval_97.5", "CI_Width", "RelativeUncertainty",
                    "ClusterVolume", "OverlapVolume", "SumProb", "P90",
                    "RatioOverlap", "DensityOverlap", "WeightedClusterContribution"]
    df = df[column_order]
    
    df.to_csv(output_csv, index=False)

    # Generate model summary file
    summary_path = output_csv.replace(".csv", "_model_summary.txt")
    generate_model_summary(model_results, summary_path)

    # Create and save the plot as a PNG file
    # Create labels combining region ID and source cluster for clarity
    labels = [f"R{row['Target_Region_ID']}:{row['Source_Cluster'][:10]}" 
              for _, row in df.iterrows()]
    
    plt.figure(figsize=(max(10, len(df) * 0.3), 6))
    plt.errorbar(x=np.arange(len(df)),
                 y=df["PosteriorMean"],
                 yerr=[df["PosteriorMean"] - df["CredibleInterval_2.5"],
                       df["CredibleInterval_97.5"] - df["PosteriorMean"]],
                 fmt='o', capsize=5)
    plt.xticks(np.arange(len(df)), labels, rotation=90)
    plt.xlabel("Target Region : Source Cluster")
    plt.ylabel("Latent Connectivity Score")
    plt.title(f"Posterior Connectivity Scores (Bayesian R²={bayesian_r2:.3f})")
    plt.tight_layout()
    plot_filename = output_csv.replace(".csv", "_plot.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    print(f"Report saved as {output_csv}")
    print(f"Plot saved as {plot_filename}")


def translate_cluster_labels(atlas_txt_path, input_csv, output_csv):
    """
    Translate target region IDs to anatomical names using a label file and re-order the report columns.

    Parameters
    ----------
    atlas_txt_path : str
        File path to the atlas text file (tab-separated) with at least 'color' (region ID)
        and 'nom_c' (anatomical name) columns.
    input_csv : str
        File path to the CSV report generated by generate_report().
    output_csv : str
        File path to save the translated report.
    """
    atlas_df = pd.read_csv(atlas_txt_path, sep="\t")
    result_df = pd.read_csv(input_csv)

    valid_regions = set(result_df["Target_Region_ID"])
    label_mapping = {color: nom_c for color, nom_c in zip(atlas_df["color"], atlas_df["nom_c"]) if color in valid_regions}

    result_df["Target_Region_Name"] = result_df["Target_Region_ID"].map(label_mapping)

    desired_order = ["Target_Region_ID", "Target_Region_Name", "Source_Cluster",
                     "PosteriorMean", "StatisticalQuality", "BayesianR2",
                     "CredibleInterval_2.5", "CredibleInterval_97.5", "CI_Width", "RelativeUncertainty",
                     "ClusterVolume", "OverlapVolume", "SumProb", "P90",
                     "RatioOverlap", "DensityOverlap", "WeightedClusterContribution"]
    result_df = result_df[[col for col in desired_order if col in result_df.columns]]

    result_df.to_csv(output_csv, index=False)
    print(f"Translated report saved as {output_csv}")

def run_connectivity_analysis(patient_disconnectome_paths, cluster_disconnectome_paths,
                              parcellation_path, output_folder, atlas_txt_path=None,
                              on_space_mismatch="error"):
    """
    Run the complete analysis for one or more patient disconnectome maps with one or more
    cluster disconnectome maps. For each patient and cluster combination, compute preserved
    connections using probability multiplication: preserved = cluster_prob × (1 - patient_prob),
    representing connections that are vulnerable in the cluster but preserved (not disconnected)
    in the patient. All cluster-region combinations are analyzed in a single Bayesian model,
    making the connectivity scores directly comparable across clusters.

    Parameters
    ----------
    patient_disconnectome_paths : list or str
        File path(s) to patient disconnectome NIfTI image(s).
    cluster_disconnectome_paths : list or str
        File path(s) to cluster disconnectome NIfTI image(s). Multiple clusters will be analyzed
        together in a single model.
    parcellation_path : str
        File path to the parcellation atlas (NIfTI image) with region labels.
    output_folder : str
        Path to the folder where output files (reports, derivatives) will be saved.
    atlas_txt_path : str, optional
        File path to the atlas text file for translating region IDs to anatomical names.
    on_space_mismatch : {'error', 'warn'}, optional
        Action to take when an affine or orientation mismatch is detected between input
        images.  ``'error'`` (default) aborts with a ``ValueError``; ``'warn'`` prints
        a warning and continues.  Shape mismatches always raise ``ValueError`` regardless
        of this setting.

    Additional Files Saved
    ------------------------
    For each patient, the following files are saved in the output_folder:
      - <patient_basename>_analysis_report.csv : CSV report with connectivity statistics.
      - translated_<patient_basename>_analysis_report.csv : (if atlas_txt_path provided)
      - <patient_basename>_<cluster_name>_preserved.nii.gz : Preserved connection maps.
      - <patient_basename>_analysis_report_plot.png : Plot of connectivity scores.
      - <patient_basename>_FAILED_diagnostic.txt : Written instead of a report when no
        cluster-region overlaps are found; contains full input diagnostics.
      - command_used.txt : The command-line call used to run the tool.
      - A 'derivatives' subfolder stores the preserved connection images.
    """
    os.makedirs(output_folder, exist_ok=True)

    command_used = " ".join(sys.argv)
    with open(os.path.join(output_folder, "command_used.txt"), "w") as f:
        f.write(command_used + "\n")

    if isinstance(patient_disconnectome_paths, str):
        patient_disconnectome_paths = [patient_disconnectome_paths]

    if isinstance(cluster_disconnectome_paths, str):
        cluster_disconnectome_paths = [cluster_disconnectome_paths]

    # Load the parcellation once so we can check its space against each patient
    parc_img = nib.load(parcellation_path)
    parc_n_regions = len(np.unique(parc_img.get_fdata().astype(int))) - 1  # exclude 0
    parc_shape = parc_img.shape[:3]

    for patient_path in patient_disconnectome_paths:
        patient_basename = Path(patient_path).stem
        print(f"Processing patient: {patient_basename}")

        patient_img = nib.load(patient_path)
        patient_data = patient_img.get_fdata()

        # --- Space check: patient vs parcellation ---
        _check_image_space(
            patient_img, patient_basename,
            parc_img, Path(parcellation_path).name,
            on_mismatch=on_space_mismatch,
        )

        # Per-cluster diagnostics collected for the failure report
        cluster_diagnostics = []

        # Process each cluster disconnectome
        all_dfs = []
        for cluster_path in cluster_disconnectome_paths:
            cluster_name = Path(cluster_path).stem
            print(f"  Processing cluster: {cluster_name}")

            cluster_img = nib.load(cluster_path)

            # --- Space check: patient vs cluster ---
            _check_image_space(
                patient_img, patient_basename,
                cluster_img, cluster_name,
                on_mismatch=on_space_mismatch,
            )

            cluster_data = cluster_img.get_fdata()

            # Compute preserved connections using probability multiplication:
            # P(preserved) = P(cluster affected) × P(NOT disconnected by patient)
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

            # Save the preserved image in the derivatives folder
            derivatives_folder = os.path.join(output_folder, "derivatives", patient_basename)
            os.makedirs(derivatives_folder, exist_ok=True)
            preserved_img = nib.Nifti1Image(preserved, affine=cluster_img.affine)
            preserved_path = os.path.join(
                derivatives_folder,
                f"{patient_basename}_{cluster_name}_preserved.nii.gz",
            )
            nib.save(preserved_img, preserved_path)
            print(f"    Saved preserved image: {preserved_path}")

            # Compute features using the preserved image and parcellation
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

        # Combine all clusters into a single dataframe
        combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        print(f"  Total cluster-region combinations: {len(combined_df)}")

        # --- Empty result: write diagnostic and skip this patient ---
        if combined_df.empty:
            diag_path = os.path.join(
                output_folder, f"{patient_basename}_FAILED_diagnostic.txt"
            )
            diag_lines = [
                "=" * 70,
                f"ANALYSIS FAILED — no cluster-region overlaps found",
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
                "     → Re-run with -sm warn to see space-check details, or inspect",
                "       the affines with: python -c \"import nibabel as nib; "
                "print(nib.load('img.nii.gz').affine)\"",
                "  4. All cluster voxel values are zero (empty cluster map).",
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

        # Run Bayesian model on combined data (returns dict with trace, R², LOO, effect sizes)
        model_results = run_bayesian_model(combined_df)

        # Generate report (includes CSV, plot, and model summary)
        report_csv = os.path.join(output_folder, f"{patient_basename}_analysis_report.csv")
        generate_report(combined_df, model_results, report_csv)

        # Generate translated report if atlas text file is provided
        if atlas_txt_path:
            translated_csv = os.path.join(
                output_folder, f"translated_{patient_basename}_analysis_report.csv"
            )
            translate_cluster_labels(atlas_txt_path, report_csv, translated_csv)

        print(f"Finished processing patient: {patient_basename}")
