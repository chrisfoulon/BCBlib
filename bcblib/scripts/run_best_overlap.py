#!/usr/bin/env python
"""
Script: run_best_overlap.py
Description: Command-line tool to run the preserved-connectivity analysis.
             For each patient disconnectome and cluster disconnectome combination,
             computes preserved connections (cluster_prob × (1 - patient_prob)),
             extracts size-normalised connectivity features per parcellation region,
             fits a robust Bayesian model (Student-t likelihood, outcome:
             log(WeightedClusterContribution)), and saves ranked CSV reports with
             ROPE-based interpretation aids.
Usage:
    python -m bcblib.scripts.run_best_overlap \\
        --patient_disconnectome patient1.nii.gz \\
        --cluster_disconnectome cluster1.nii.gz cluster2.nii.gz \\
        --parcellation atlas.nii.gz \\
        --output_folder output_dir/ \\
        [--atlas_txt labels.txt] \\
        [--min_ratio_overlap 0.02] \\
        [--min_overlap_voxels 5] \\
        [--rope_high -3.0]
"""

import argparse
from bcblib.tools.best_overlap import run_connectivity_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run preserved-connectivity analysis using patient disconnectome maps, "
            "cluster disconnectome(s), and a parcellation atlas. "
            "Multiple cluster disconnectomes are analysed together in a single "
            "Bayesian model so that posterior scores are directly comparable."
        )
    )
    parser.add_argument(
        "--patient_disconnectome", nargs="+", required=True,
        help="Path(s) to patient disconnectome NIfTI image(s) (probability maps, 0–1).",
    )
    parser.add_argument(
        "--cluster_disconnectome", nargs="+", required=True,
        help=(
            "Path(s) to cluster disconnectome NIfTI image(s). "
            "Multiple clusters are analysed together in a single model."
        ),
    )
    parser.add_argument(
        "--parcellation", required=True,
        help="Path to the parcellation atlas NIfTI image (integer region labels).",
    )
    parser.add_argument(
        "--output_folder", required=True,
        help="Folder where all output files will be saved.",
    )
    parser.add_argument(
        "--atlas_txt",
        help=(
            "Optional: path to the atlas TSV file for translating region IDs "
            "to anatomical names (columns: 'color', 'nom_c')."
        ),
    )
    parser.add_argument(
        "-sm", "--on-space-mismatch",
        choices=["error", "warn"], default="error",
        dest="on_space_mismatch",
        help=(
            "Action when an affine or orientation mismatch is detected between "
            "input images. 'error' (default) aborts with a clear explanation; "
            "'warn' prints a warning and continues. Shape mismatches always abort."
        ),
    )
    parser.add_argument(
        "--min_ratio_overlap", type=float, default=None,
        metavar="FLOAT",
        help=(
            "Size-normalised detection filter: exclude region-cluster pairs where "
            "fewer than this fraction of region voxels show preserved connections "
            "(RatioOverlap = OverlapVolume / ClusterVolume). "
            "Example: 0.01 requires at least 1%% of region voxels to overlap. "
            "Because the threshold is relative to region size, it does not "
            "penalise small regions unfairly. "
            "DEFAULT: off (None). "
            "WARNING: set this value based on anatomical/spatial-resolution "
            "grounds BEFORE inspecting results. Choosing a threshold to improve "
            "the statistics after the fact is equivalent to p-hacking."
        ),
    )
    parser.add_argument(
        "--min_overlap_voxels", type=int, default=None,
        metavar="INT",
        help=(
            "Absolute detection floor: exclude region-cluster pairs with fewer "
            "than this many overlap voxels regardless of region size. "
            "Guards against single-voxel partial-volume artefacts "
            "(e.g. --min_overlap_voxels 5 for 2 mm isotropic data). "
            "DEFAULT: off (None). "
            "WARNING: same caution as --min_ratio_overlap applies. "
            "Both filters are applied with AND logic when both are set."
        ),
    )
    parser.add_argument(
        "--rope_high", type=float, default=None,
        metavar="FLOAT",
        help=(
            "Upper bound of the negligible zone on the log-WCC scale for ROPE "
            "classification (Kruschke, 2018). Pairs whose entire 95%% credible "
            "interval falls at or below rope_high are labelled 'Negligible'. "
            "DEFAULT: None → computed automatically as the 25th percentile of the "
            "raw observed log(WCC) distribution (pre-model, no dependence on "
            "posterior outputs)."
        ),
    )
    return parser.parse_args()


def main_script():
    args = parse_args()
    run_connectivity_analysis(
        args.patient_disconnectome,
        args.cluster_disconnectome,
        args.parcellation,
        args.output_folder,
        args.atlas_txt,
        on_space_mismatch=args.on_space_mismatch,
        min_ratio_overlap=args.min_ratio_overlap,
        min_overlap_voxels=args.min_overlap_voxels,
        rope_high=args.rope_high,
    )


if __name__ == "__main__":
    main_script()
