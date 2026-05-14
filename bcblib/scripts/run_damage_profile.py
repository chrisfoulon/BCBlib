#!/usr/bin/env python
"""CLI for damage_profile: compute overlap statistics between a subject map
and one or more brain atlases.

Usage (custom atlas directory)::

    bcb-damage-profile --map lesion.nii.gz --atlas /path/to/atlas --name jhu

Usage (preset atlas)::

    bcb-damage-profile --map lesion.nii.gz --preset rojkova --output-dir ./results

Usage (multiple atlases)::

    bcb-damage-profile --map lesion.nii.gz \\
        --atlas /path/to/dir --name cortical \\
        --atlas /path/to/4d.nii.gz --name subcortical --threshold 0.2

Multiple ``--atlas``/``--name`` pairs are matched positionally.  Each
``--threshold`` value is matched positionally to the corresponding atlas;
if fewer thresholds than atlases are given, 0.0 is used for the remainder.
"""

import argparse
import sys
from pathlib import Path

from bcblib.tools.damage_profile import AtlasSpec, damage_profile, get_preset_atlas
from bcblib.tools.damage_profile._atlas_manager import list_preset_atlases


def parse_args(argv=None):
    """Parse command-line arguments for ``bcb-damage-profile``.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="bcb-damage-profile",
        description=(
            "Compute per-region overlap statistics between a subject map "
            "(lesion or disconnectome) and one or more brain atlases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--map", required=True, metavar="PATH",
        help="Path to the subject NIfTI map (lesion or disconnectome).",
    )

    # Atlas group — custom path
    atlas_group = parser.add_argument_group("Custom atlas (repeatable)")
    atlas_group.add_argument(
        "--atlas", action="append", metavar="PATH", default=[],
        dest="atlas_paths",
        help=(
            "Path to atlas: a directory of NIfTI files, a 4D NIfTI, or a "
            "label NIfTI.  Repeat for multiple atlases."
        ),
    )
    atlas_group.add_argument(
        "--name", action="append", metavar="NAME", default=[],
        dest="atlas_names",
        help="Short name for the atlas (matches positionally to --atlas).",
    )
    atlas_group.add_argument(
        "--threshold", action="append", type=float, metavar="FLOAT", default=[],
        dest="atlas_thresholds",
        help=(
            "Minimum weight value retained for this atlas (default 0.0). "
            "Matches positionally to --atlas."
        ),
    )
    atlas_group.add_argument(
        "--label-file", action="append", metavar="PATH", default=[],
        dest="atlas_label_files",
        help="Label file for this atlas (matches positionally to --atlas).",
    )

    # Preset atlas
    preset_group = parser.add_argument_group("Preset atlas")
    preset_group.add_argument(
        "--preset", action="append", metavar="NAME", default=[],
        dest="presets",
        help=(
            f"Named preset atlas.  Available: "
            f"{', '.join(list_preset_atlases())}.  Repeat for multiple presets."
        ),
    )
    preset_group.add_argument(
        "--preset-path", action="append", metavar="PATH", default=[],
        dest="preset_paths",
        help=(
            "Explicit local path for the corresponding --preset "
            "(matches positionally).  Skips download if provided."
        ),
    )
    preset_group.add_argument(
        "--assume-yes", action="store_true", default=False,
        help="Skip the download consent prompt for preset atlases.",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", metavar="DIR", default=None,
        dest="output_dir",
        help=(
            "Directory to write CSV results to.  One file per atlas: "
            "<name>_damage_profile.csv.  Defaults to the current directory."
        ),
    )
    output_group.add_argument(
        "--min-overlap-voxels", type=int, default=1,
        dest="min_overlap_voxels",
        help="Minimum non-zero overlapping voxels to include a region. Default: 1.",
    )
    output_group.add_argument(
        "--on-space-mismatch", choices=["error", "warn"], default="error",
        dest="on_space_mismatch",
        help=(
            "Behaviour when a same-family space mismatch is detected: "
            "'error' (default) or 'warn' (resample and continue)."
        ),
    )

    args = parser.parse_args(argv)

    # Validate: at least one atlas or preset
    if not args.atlas_paths and not args.presets:
        parser.error("Provide at least one --atlas or --preset.")

    # Validate: each --atlas must have a matching --name
    if len(args.atlas_paths) != len(args.atlas_names):
        parser.error(
            f"Got {len(args.atlas_paths)} --atlas argument(s) but "
            f"{len(args.atlas_names)} --name argument(s). They must match."
        )

    return args


def build_atlas_specs(args):
    """Construct an :class:`AtlasSpec` list from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Output of :func:`parse_args`.

    Returns
    -------
    list of AtlasSpec
        One entry per ``--atlas``/``--name`` pair, in order.
    """
    specs = []

    # Custom atlases
    for i, (path, name) in enumerate(zip(args.atlas_paths, args.atlas_names)):
        threshold = args.atlas_thresholds[i] if i < len(args.atlas_thresholds) else 0.0
        label_file = args.atlas_label_files[i] if i < len(args.atlas_label_files) else None
        specs.append(AtlasSpec(
            source=path,
            name=name,
            threshold=threshold,
            label_file=label_file,
        ))

    return specs


def main(argv=None):
    """Entry point for the ``bcb-damage-profile`` CLI.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = parse_args(argv)

    output_dir = args.output_dir or "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    specs = build_atlas_specs(args)

    # Preset atlases — resolve to AtlasSpec via get_preset_atlas then wrap
    # (get_preset_atlas returns a dict; we need an AtlasSpec for damage_profile)
    # Strategy: call damage_profile directly with preset dicts for presets.
    # For a clean interface, we call get_preset_atlas and pass the resolved
    # cache path back as a spec.
    preset_specs = []
    for i, preset_name in enumerate(args.presets):
        explicit_path = args.preset_paths[i] if i < len(args.preset_paths) else None
        try:
            # Resolve to path: if explicit_path given use it, else let manager
            # find/download to cache and return the spec source.
            from bcblib.tools.damage_profile._atlas_manager import (
                get_atlas_dir, PRESET_ATLASES,
            )
            info = PRESET_ATLASES[preset_name]
            if explicit_path:
                source = explicit_path
            else:
                cache = get_atlas_dir() / preset_name
                # Trigger download if needed
                get_preset_atlas(preset_name, assume_yes=args.assume_yes)
                source = str(cache / info.nifti_path) if info.nifti_path else str(cache)
            preset_specs.append(AtlasSpec(
                source=source,
                name=preset_name,
                space=info.space,
            ))
        except KeyError:
            print(
                f"Error: unknown preset '{preset_name}'. "
                f"Available: {', '.join(list_preset_atlases())}",
                file=sys.stderr,
            )
            sys.exit(1)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    all_specs = specs + preset_specs

    results = damage_profile(
        args.map,
        all_specs,
        min_overlap_voxels=args.min_overlap_voxels,
        on_space_mismatch=args.on_space_mismatch,
        output_dir=output_dir,
    )

    stats_out = Path(output_dir) / "subject_map_stats.csv"
    print(f"  subject stats → {stats_out}")

    for name, df in results.items():
        if name == "_subject_map_stats":
            continue
        out = Path(output_dir) / f"{name}_damage_profile.csv"
        print(f"  {name}: {len(df)} regions → {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
