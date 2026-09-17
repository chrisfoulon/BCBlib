"""Post-hoc repair of pwll_normalised/continuous_dice in existing CSV output.

Fixes output written by bcblib<=0.7.1 (see
``bcblib.tools.damage_profile._stats.correct_probabilistic_metrics``)
without recomputing anything from the source lesion/disconnectome NIfTIs —
only previously-written CSVs are read and rewritten.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from bcblib.tools.damage_profile._stats import correct_probabilistic_metrics


def fix_existing_outputs(output_dir) -> Dict[Path, int]:
    """Repair pwll_normalised/continuous_dice in every affected CSV under *output_dir*.

    Recursively scans for ``*.csv`` files and applies
    :func:`correct_probabilistic_metrics` to each. Files that don't have the
    required columns (label-atlas overlap CSVs, streamline-ratio CSVs, or
    anything not produced by this pipeline) are left untouched — not even
    reopened for writing. Idempotent: safe to run on output already fixed by
    a previous call, or already correct because it came from bcblib>=0.7.2.

    Parameters
    ----------
    output_dir : str or Path
        Root of a ``bcb-lesion-features`` output directory (or any directory
        tree containing its CSVs).

    Returns
    -------
    dict[Path, int]
        Only the files that actually had a value changed, mapped to the
        number of rows corrected in each.
    """
    output_dir = Path(output_dir)
    changed: Dict[Path, int] = {}

    for csv_path in sorted(output_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue

        fixed_df, n_changed = correct_probabilistic_metrics(df)
        if n_changed > 0:
            fixed_df.to_csv(csv_path, index=False)
            changed[csv_path] = n_changed

    return changed
