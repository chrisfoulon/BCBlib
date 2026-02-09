#!/usr/bin/env python
"""
Script: run_connectivity.py
Description: Command-line tool to run connectivity analysis. For each patient disconnectome and
             cluster disconnectome combination, it computes preserved connections using probability
             multiplication: cluster_prob × (1 - patient_prob), computes connectivity features using
             a parcellation atlas, runs a Bayesian model on all combinations together, and saves
             the output reports.
Usage:
    python run_connectivity.py --patient_disconnectome patient1.nii patient2.nii --cluster_disconnectome cluster1.nii cluster2.nii cluster3.nii --parcellation atlas.nii --output_folder output_dir [--atlas_txt labels.txt]
"""

import argparse
from bcblib.tools.best_overlap import run_connectivity_analysis

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run connectivity analysis using patient disconnectome maps, cluster disconnectome(s), and a parcellation atlas. Multiple cluster disconnectomes are analyzed together in a single model."
    )
    parser.add_argument("--patient_disconnectome", nargs="+", required=True,
                        help="File path(s) to patient disconnectome NIfTI image(s).")
    parser.add_argument("--cluster_disconnectome", nargs="+", required=True,
                        help="File path(s) to cluster disconnectome NIfTI image(s). Multiple clusters will be analyzed together.")
    parser.add_argument("--parcellation", required=True,
                        help="File path to the parcellation atlas (NIfTI image) with region labels.")
    parser.add_argument("--output_folder", required=True,
                        help="Folder where output files will be saved.")
    parser.add_argument("--atlas_txt",
                        help="Optional: File path to the atlas text file for translating region IDs to anatomical names.")
    return parser.parse_args()

def main_script():
    args = parse_args()
    run_connectivity_analysis(args.patient_disconnectome,
                              args.cluster_disconnectome,
                              args.parcellation,
                              args.output_folder,
                              args.atlas_txt)

if __name__ == "__main__":
    main_script()
