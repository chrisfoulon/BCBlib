"""
Module: best_overlap (formerly connectivity)
Description: Provides functions to process disconnectome maps by subtracting a patient’s
             disconnectome from a cluster disconnectome, thresholding negative values,
             computing connectivity features using a parcellation atlas, running a Bayesian
             model to obtain latent connectivity scores, and generating CSV reports (including
             uncertainty measures). Optionally, cluster IDs can be translated to anatomical names
             using a label file.
"""

import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def compute_cluster_features(preserved_map, parcellation_path):
    """
    Compute connectivity features for each cluster using a preserved connection map
    and a parcellation atlas.

    Parameters
    ----------
    preserved_map : numpy.ndarray
        A 3D array representing the preserved connections (cluster disconnectome minus
        patient disconnectome, thresholded so that negative values are 0).
    parcellation_path : str
        File path to the parcellation atlas (NIfTI image) where voxels are labeled by cluster number.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing computed features for each cluster. Columns include:
          - Cluster: Cluster ID (integer)
          - ClusterVolume: Total number of voxels in the cluster.
          - OverlapVolume: Number of voxels in the cluster with preserved connections (value > 0).
          - SumProb: Sum of preserved connection values within the cluster.
          - P90: 90th percentile of the preserved connection values in the cluster.
          - RatioOverlap: OverlapVolume divided by ClusterVolume.
          - DensityOverlap: SumProb divided by OverlapVolume.
          - WeightedClusterContribution: SumProb divided by ClusterVolume.
    """
    parc_img = nib.load(parcellation_path)
    parc_data = parc_img.get_fdata().astype(int)

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
                      columns=["Cluster", "ClusterVolume", "OverlapVolume", "SumProb", "P90",
                               "RatioOverlap", "DensityOverlap", "WeightedClusterContribution"])
    df = df[df["OverlapVolume"] > 0]
    return df

def run_bayesian_model(df):
    """
    Build and sample from a Bayesian regression model to predict a composite connectivity measure.
    The model uses the features computed from the preserved connections and regresses out the effect
    of cluster size via a log-transformation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cluster features (from compute_cluster_features).

    Returns
    -------
    trace : arviz.InferenceData
        Posterior samples from the Bayesian model.

    Model Details
    -------------
    - Outcome: log(SumProb+1), reflecting both the extent and intensity of preserved connections.
    - Predictors include: log(ClusterVolume+1), P90, RatioOverlap, DensityOverlap, and WeightedClusterContribution.
    - A deterministic variable 'mu_det' (latent connectivity score) is defined for ranking.
    """
    with pm.Model() as model:
        beta0    = pm.Normal("beta0", mu=0, sigma=10)
        beta_cv  = pm.Normal("beta_cv", mu=0, sigma=10)
        beta_p90 = pm.Normal("beta_p90", mu=0, sigma=10)
        beta_ro  = pm.Normal("beta_ro", mu=0, sigma=10)
        beta_de  = pm.Normal("beta_de", mu=0, sigma=10)
        beta_wc  = pm.Normal("beta_wc", mu=0, sigma=10)

        log_cluster_vol = np.log(df["ClusterVolume"] + 1)
        y = np.log(df["SumProb"] + 1)

        mu = (beta0 + beta_cv * log_cluster_vol + beta_p90 * df["P90"] +
              beta_ro * df["RatioOverlap"] + beta_de * df["DensityOverlap"] +
              beta_wc * df["WeightedClusterContribution"])
        mu_det = pm.Deterministic("mu_det", mu)

        sigma = pm.HalfNormal("sigma", sigma=5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(2000, return_inferencedata=True)
    return trace


def generate_report(df, trace, output_csv):
    """
    Extract posterior summaries for the latent connectivity score, rank clusters, and
    generate a CSV report that includes uncertainty measures and a relative uncertainty column.
    The plot is saved to disk.
    """
    # Check if any clusters were found
    if df.empty:
        print("No clusters with preserved connections were found. Skipping report generation.")
        return

    mu_samples = trace.posterior["mu_det"]
    combined_samples = mu_samples.stack(sample=("chain", "draw")).values
    posterior_mean = combined_samples.mean(axis=1)
    ci_lower = np.percentile(combined_samples, 2.5, axis=1)
    ci_upper = np.percentile(combined_samples, 97.5, axis=1)
    ci_width = ci_upper - ci_lower
    relative_uncertainty = ci_width / posterior_mean

    df["PosteriorMean"] = posterior_mean
    df["CredibleInterval_2.5"] = ci_lower
    df["CredibleInterval_97.5"] = ci_upper
    df["CI_Width"] = ci_width
    df["RelativeUncertainty"] = relative_uncertainty

    quantiles = np.percentile(posterior_mean, [20, 40, 60, 80])

    def interpret(score):
        if score >= quantiles[3]:
            return "Best"
        elif score >= quantiles[2]:
            return "Good"
        elif score >= quantiles[1]:
            return "Decent"
        elif score >= quantiles[0]:
            return "Passable"
        else:
            return "Mediocre"

    df["Interpretation"] = [interpret(score) for score in posterior_mean]

    df = df.sort_values(by="PosteriorMean", ascending=False)
    df.to_csv(output_csv, index=False)

    # Create and save the plot as a PNG file
    plt.errorbar(x=np.arange(len(df)),
                 y=df["PosteriorMean"],
                 yerr=[df["PosteriorMean"] - df["CredibleInterval_2.5"],
                       df["CredibleInterval_97.5"] - df["PosteriorMean"]],
                 fmt='o', capsize=5)
    plt.xticks(np.arange(len(df)), df["Cluster"], rotation=90)
    plt.xlabel("Cluster (ID)")
    plt.ylabel("Latent Connectivity Score")
    plt.title("Posterior Connectivity Scores with 95% Credible Intervals")
    plt.tight_layout()
    plot_filename = output_csv.replace(".csv", "_plot.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    print(f"Report saved as {output_csv}")
    print(f"Plot saved as {plot_filename}")


def translate_cluster_labels(atlas_txt_path, input_csv, output_csv):
    """
    Translate cluster IDs to anatomical names using a label file and re-order the report columns.

    Parameters
    ----------
    atlas_txt_path : str
        File path to the atlas text file (tab-separated) with at least 'color' (cluster ID)
        and 'nom_c' (anatomical name) columns.
    input_csv : str
        File path to the CSV report generated by generate_report().
    output_csv : str
        File path to save the translated report.
    """
    atlas_df = pd.read_csv(atlas_txt_path, sep="\t")
    result_df = pd.read_csv(input_csv)

    valid_clusters = set(result_df["Cluster"])
    label_mapping = {color: nom_c for color, nom_c in zip(atlas_df["color"], atlas_df["nom_c"]) if color in valid_clusters}

    result_df["ClusterName"] = result_df["Cluster"].map(label_mapping)

    desired_order = ["Cluster", "ClusterName", "PosteriorMean", "Interpretation",
                     "ClusterVolume", "OverlapVolume", "SumProb", "P90",
                     "RatioOverlap", "DensityOverlap", "WeightedClusterContribution",
                     "CredibleInterval_2.5", "CredibleInterval_97.5", "CI_Width", "RelativeUncertainty"]
    result_df = result_df[[col for col in desired_order if col in result_df.columns]]

    result_df.to_csv(output_csv, index=False)
    print(f"Translated report saved as {output_csv}")

def run_connectivity_analysis(patient_disconnectome_paths, cluster_disconnectome_path,
                              parcellation_path, output_folder, atlas_txt_path=None):
    """
    Run the complete analysis for one or more patient disconnectome maps.
    For each patient, subtract the patient disconnectome from the cluster disconnectome
    (thresholding negative values to 0 to keep only preserved connections), compute connectivity features
    using the parcellation atlas, run a Bayesian model to compute a latent connectivity score, and save output
    files in the output folder.

    Parameters
    ----------
    patient_disconnectome_paths : list or str
        File path(s) to patient disconnectome NIfTI image(s).
    cluster_disconnectome_path : str
        File path to the cluster disconnectome NIfTI image.
    parcellation_path : str
        File path to the parcellation atlas (NIfTI image) with cluster labels.
    output_folder : str
        Path to the folder where output files (reports, derivatives) will be saved.
    atlas_txt_path : str, optional
        File path to the atlas text file for translating cluster IDs to anatomical names.

    Additional Files Saved
    ------------------------
    For each patient, the following files are saved in the output_folder (or a subfolder for the patient):
      - <patient_basename>_analysis_report.csv : CSV report with connectivity statistics.
      - translated_<patient_basename>_analysis_report.csv : (if atlas_txt_path provided) CSV report with cluster names.
      - <patient_basename>_preserved.nii.gz : The subtracted (preserved connections) NIfTI image.
      - <patient_basename>_analysis_report_plot.png : Plot of connectivity scores.
      - command_used.txt : A text file containing the command-line call used to run the tool.
      - A 'derivatives' subfolder is created to store additional outputs.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load cluster disconnectome map once
    cluster_img = nib.load(cluster_disconnectome_path)
    cluster_data = cluster_img.get_fdata()

    import sys
    command_used = " ".join(sys.argv)
    with open(os.path.join(output_folder, "command_used.txt"), "w") as f:
        f.write(command_used + "\n")

    if isinstance(patient_disconnectome_paths, str):
        patient_disconnectome_paths = [patient_disconnectome_paths]

    for patient_path in patient_disconnectome_paths:
        patient_basename = Path(patient_path).stem
        print(f"Processing patient: {patient_basename}")

        patient_img = nib.load(patient_path)
        patient_data = patient_img.get_fdata()

        # Subtract and threshold: preserved = cluster - patient; negative values become 0
        preserved = cluster_data - patient_data
        preserved[preserved < 0] = 0

        # Save the preserved image in the derivatives folder
        derivatives_folder = os.path.join(output_folder, "derivatives", patient_basename)
        os.makedirs(derivatives_folder, exist_ok=True)
        preserved_img = nib.Nifti1Image(preserved, affine=cluster_img.affine)
        preserved_path = os.path.join(derivatives_folder, f"{patient_basename}_preserved.nii.gz")
        nib.save(preserved_img, preserved_path)
        print(f"Saved preserved image: {preserved_path}")

        # Compute features using the preserved image and parcellation
        df = compute_cluster_features(preserved, parcellation_path)

        # Run Bayesian model
        trace = run_bayesian_model(df)

        # Generate report
        report_csv = os.path.join(output_folder, f"{patient_basename}_analysis_report.csv")
        generate_report(df, trace, report_csv)

        # Generate translated report if atlas text file is provided
        if atlas_txt_path:
            translated_csv = os.path.join(output_folder, f"translated_{patient_basename}_analysis_report.csv")
            translate_cluster_labels(atlas_txt_path, report_csv, translated_csv)

        print(f"Finished processing patient: {patient_basename}")
