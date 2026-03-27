#!/usr/bin/env python
"""
Script: run_dataset_splitting.py
Description: CLI for balanced dataset splitting via Monte Carlo permutation
             search. Reads a CSV file with one row per subject, runs
             permutation_balanced_splits, and writes the same CSV with an
             added 'fold' column (0-indexed).

Usage:
    bcb-dataset-split --input subjects.csv --group-col has_chronic \\
        --covariate-cols acute_volume chronic_volume --output splits.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bcblib.tools.dataset_splitting import permutation_balanced_splits


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Balanced dataset splitting via Monte Carlo permutation search. "
            "Reads a CSV with one row per subject, assigns each subject to a "
            "fold, and writes the result as a CSV with an added 'fold' column."
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file. Must contain the group column and all "
             "covariate columns."
    )
    parser.add_argument(
        "--group-col", required=True, dest="group_col",
        help="Name of the CSV column to use as the categorical grouping "
             "variable."
    )
    parser.add_argument(
        "--covariate-cols", required=True, nargs="+", dest="covariate_cols",
        help="One or more column names to use as continuous covariates for "
             "Kruskal-Wallis scoring."
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, dest="n_splits",
        help="Number of folds. Default: 5."
    )
    parser.add_argument(
        "--n-permutations", type=int, default=50000, dest="n_permutations",
        help="Number of random permutations to try. Default: 50000."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the output CSV file (input CSV + 'fold' column)."
    )
    parser.add_argument(
        "--no-strict", action="store_true", dest="no_strict",
        help="Disable the strict group-size check. By default, an error is "
             "raised if any group has fewer subjects than --n-splits, with a "
             "report of every offending group and its count. Use --no-strict "
             "to skip this check (underfilled folds will score 0.0 and will "
             "not be selected as the best split)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    all_cols = [args.group_col] + args.covariate_cols
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"Error: column(s) not found in input CSV: {missing}",
              file=sys.stderr)
        sys.exit(1)

    groups = df[args.group_col].to_numpy()
    covariates = {
        col: df[col].to_numpy(dtype=float) for col in args.covariate_cols
    }

    fold_indices, score, report = permutation_balanced_splits(
        groups=groups,
        covariates=covariates,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        seed=args.seed,
        strict=not args.no_strict,
    )

    fold_col = np.empty(len(df), dtype=int)
    for fold_num, indices in enumerate(fold_indices):
        for idx in indices:
            fold_col[idx] = fold_num

    df['fold'] = fold_col
    df.to_csv(args.output, index=False)

    report_path = Path(args.output).with_suffix('').as_posix() + '_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    n_improved = len(report['search']['convergence'])
    best_at = report['search']['best_found_at_permutation']
    print(f"Best split score (minimax KW p-value): {score:.6f}")
    print(f"  Best found at permutation {best_at}/{args.n_permutations} "
          f"({n_improved} improvements)")
    print(f"Output CSV:    {args.output}")
    print(f"Report JSON:   {report_path}")


if __name__ == "__main__":
    main()
