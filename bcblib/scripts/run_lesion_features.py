"""CLI: bcb-lesion-features — extract lesion and disconnectome features."""

import argparse
import sys
from pathlib import Path


def _build_parser():
    p = argparse.ArgumentParser(
        prog="bcb-lesion-features",
        description=(
            "Stage 2 of the lesion-features pipeline: compute atlas overlap "
            "features for lesion masks and disconnectome maps."
        ),
    )
    p.add_argument(
        "--prep-dir", required=True, metavar="PATH",
        help="Output directory from bcb-lf-preprocess",
    )
    p.add_argument(
        "--output-dir", default="./lesion_features", metavar="DIR",
        help="Output directory for feature CSVs and TSVs (default: ./lesion_features)",
    )
    # Custom atlas arguments (repeatable)
    p.add_argument(
        "--atlas", action="append", default=[], metavar="PATH",
        help="Path to a custom atlas (repeatable; requires matching --name)",
    )
    p.add_argument(
        "--name", action="append", default=[], metavar="NAME",
        help="Short identifier for the corresponding --atlas (repeatable)",
    )
    p.add_argument(
        "--threshold", action="append", type=float, default=[], metavar="F",
        help="Minimum weight for the corresponding --atlas (repeatable, default 0)",
    )
    p.add_argument(
        "--label-file", action="append", default=[], metavar="PATH",
        help="Label file for the corresponding --atlas (repeatable, optional)",
    )
    # Preset atlases
    p.add_argument(
        "--preset", action="append", default=[], metavar="NAME",
        help="Add a preset atlas by key (repeatable; see bcb-damage-profile --list)",
    )
    p.add_argument(
        "--ebrains", action="store_true",
        help="Use the full EBRAINS default atlas set",
    )
    p.add_argument(
        "--assume-yes", action="store_true",
        help="Skip download consent prompts for preset atlases",
    )
    p.add_argument(
        "--min-overlap-voxels", type=int, default=1, metavar="N",
        help="Minimum voxel overlap to include a region (default: 1)",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip subjects whose output already exists",
    )
    p.add_argument(
        "--tdi-dir", default=None, metavar="PATH",
        help=(
            "Directory containing the private tdi.py script and "
            "tdi_map_1mm.nii atlas (default: $TDI_DIR or /opt/tdi). "
            "TDI is skipped with a warning if not found."
        ),
    )
    return p


def _parse_atlas_specs(args):
    """Build a list of AtlasSpec from CLI arguments."""
    from bcblib.tools.damage_profile import AtlasSpec

    if args.atlas and len(args.atlas) != len(args.name):
        print(
            "ERROR: --atlas and --name must be provided in matching pairs.",
            file=sys.stderr,
        )
        sys.exit(1)

    specs = []
    for i, (path, name) in enumerate(zip(args.atlas, args.name)):
        threshold = args.threshold[i] if i < len(args.threshold) else 0.0
        label_file = args.label_file[i] if i < len(args.label_file) else None
        specs.append(AtlasSpec(
            source=path, name=name, threshold=threshold, label_file=label_file,
        ))
    return specs


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    prep_dir = Path(args.prep_dir)
    output_dir = Path(args.output_dir)

    if not prep_dir.exists():
        print(f"ERROR: prep-dir does not exist: {prep_dir}", file=sys.stderr)
        sys.exit(1)

    # Assemble atlas list
    atlas_specs = _parse_atlas_specs(args)

    if args.ebrains:
        from bcblib.tools.lesion_features._pipeline import get_ebrains_atlas_specs
        atlas_specs = get_ebrains_atlas_specs(assume_yes=args.assume_yes) + atlas_specs

    for key in args.preset:
        from bcblib.tools.damage_profile._atlas_manager import get_preset_atlas, PRESET_ATLASES
        from bcblib.tools.damage_profile import AtlasSpec
        get_preset_atlas(key, assume_yes=args.assume_yes)
        info = PRESET_ATLASES[key]
        from bcblib.tools.damage_profile._atlas_manager import get_atlas_dir
        cache = get_atlas_dir() / key
        source = str(cache / info.nifti_path) if info.nifti_path else str(cache)
        atlas_specs.append(AtlasSpec(source=source, name=key))

    if not atlas_specs:
        print(
            "ERROR: no atlases specified. Use --atlas/--name, --preset, or --ebrains.",
            file=sys.stderr,
        )
        sys.exit(1)

    force = not args.skip_existing
    from bcblib.tools.lesion_features._pipeline import extract_features_batch
    print(f"Extracting features from {prep_dir} → {output_dir}")
    results = extract_features_batch(
        prep_dir, atlas_specs, output_dir, force=force, tdi_dir=args.tdi_dir,
    )
    print(f"Processed {len(results)} subject(s).")


if __name__ == "__main__":
    main()
