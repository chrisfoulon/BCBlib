"""CLI: bcb-lf-preprocess — normalise lesions and compute disconnectomes."""

import argparse
import sys
from pathlib import Path

_BIDS_SUFFIX = "*_label-lesion_mask.nii.gz"
_FLAT_SUFFIX = "*_lesion_mask.nii.gz"


def _build_parser():
    p = argparse.ArgumentParser(
        prog="bcb-lf-preprocess",
        description=(
            "Stage 1 of the lesion-features pipeline: normalise lesion masks to "
            "MNI152NLin6Asym 1 mm and compute disconnectomes with BCBToolKit."
        ),
    )
    p.add_argument(
        "--bids-dir", required=True, metavar="PATH",
        help=(
            "Input directory.  With the default BIDS layout, expects "
            "sub-*/anat/*_label-lesion_mask.nii.gz.  Use --flat for "
            "pipelines that place files directly under sub-*/ (e.g. StrokeBrain)."
        ),
    )
    p.add_argument(
        "--output-dir", default="./lesion_features_prep", metavar="DIR",
        help="Output BIDS derivatives directory (default: ./lesion_features_prep)",
    )
    p.add_argument(
        "--bcbtoolkit", default=None, metavar="PATH",
        help="Path to BCBToolKit directory containing run_disco.sh",
    )
    p.add_argument(
        "--tracks-dir", default=None, metavar="PATH",
        help="Path to tractography atlas directory (-T flag for run_disco.sh)",
    )
    p.add_argument(
        "--ncores", type=int, default=None, metavar="N",
        help="Number of parallel cores for run_disco.sh",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip subjects whose output already exists",
    )
    p.add_argument(
        "--tmpdir", default=None, metavar="PATH",
        help=(
            "Directory for run_disco.sh intermediate files (-w flag). "
            "Defaults to $TMPDIR/bcb_disco_<PID> or /tmp. "
            "Set this on systems where /tmp is restricted (e.g. JupyterHub/HPC)."
        ),
    )
    p.add_argument(
        "--flat", action="store_true",
        help=(
            "Input uses a flat layout: lesion files sit directly under sub-*/ "
            "with no anat/ subfolder (e.g. StrokeBrain output).  "
            "Changes the default --lesion-suffix to '*_lesion_mask.nii.gz'."
        ),
    )
    p.add_argument(
        "--lesion-suffix", default=None, metavar="GLOB",
        help=(
            "Glob pattern for lesion mask filenames inside each subject directory. "
            "Defaults to '*_label-lesion_mask.nii.gz' (BIDS) or "
            "'*_lesion_mask.nii.gz' when --flat is set. "
            "Override if your project uses a different naming convention."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    bids_dir = Path(args.bids_dir)
    output_dir = Path(args.output_dir)
    force = not args.skip_existing

    # Resolve lesion suffix: explicit > flat default > BIDS default
    if args.lesion_suffix is not None:
        suffix = args.lesion_suffix
    elif args.flat:
        suffix = _FLAT_SUFFIX
    else:
        suffix = _BIDS_SUFFIX

    if not bids_dir.exists():
        print(f"ERROR: bids-dir does not exist: {bids_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        from bcblib.tools.lesion_features._bids import iter_bids_lesions, iter_flat_lesions
        iterator = (
            iter_flat_lesions(bids_dir, suffix=suffix)
            if args.flat
            else iter_bids_lesions(bids_dir, suffix=suffix)
        )
        subjects = list(iterator)
        mode = "flat" if args.flat else "BIDS"
        print(f"Would preprocess {len(subjects)} lesion(s) [{mode} mode, suffix={suffix!r}]:")
        for sub_id, ses_id, path in subjects:
            ses_part = f" ses-{ses_id}" if ses_id else ""
            print(f"  sub-{sub_id}{ses_part}: {path}")
        return

    from bcblib.tools.lesion_features._pipeline import preprocess_batch
    from bcblib.tools.lesion_features._disco import find_bcbtoolkit, run_disco_batch

    mode = "flat" if args.flat else "BIDS"
    print(f"Preprocessing lesions [{mode} mode] → {output_dir}")
    results = preprocess_batch(
        bids_dir, output_dir, force=force, suffix=suffix, flat=args.flat,
    )
    print(f"Preprocessed {len(results)} subject(s).")

    try:
        kit = find_bcbtoolkit(args.bcbtoolkit)
    except FileNotFoundError as e:
        print(f"WARNING: {e}\nSkipping disconnectome computation.", file=sys.stderr)
        return

    # Flatten all normalised lesions (always BIDS-named in the prep dir) into a
    # tmp dir for run_disco.sh, then move the outputs back into BIDS structure.
    from bcblib.tools.lesion_features._bids import iter_bids_lesions
    from bcblib.tools.lesion_features._disco import predict_disco_output
    import shutil

    lesion_dir = output_dir / "_tmp_lesions_for_disco"
    lesion_dir.mkdir(parents=True, exist_ok=True)
    disco_flat = output_dir / "_tmp_disco_flat"
    sub_map = {}  # expected_disco_stem → lesion_dir
    try:
        for sub_id, ses_id, lesion_path in iter_bids_lesions(output_dir, subdir="lesion"):
            shutil.copy2(str(lesion_path), str(lesion_dir / lesion_path.name))
            expected = predict_disco_output(lesion_path, disco_flat)
            stem = expected.name.replace(".nii.gz", "")
            sub_map[stem] = lesion_path.parent

        n = len(list(lesion_dir.glob("*.nii.gz")))
        print(f"Running disconnectome computation for {n} subject(s)...")
        run_disco_batch(
            lesion_dir, disco_flat, kit,
            ncores=args.ncores, tracks_dir=args.tracks_dir,
            tmpdir=args.tmpdir,
        )

        moved = 0
        for disco_file in sorted(disco_flat.glob("*disconnectome.nii.gz")):
            stem = disco_file.name.replace(".nii.gz", "")
            if stem in sub_map:
                dest = sub_map[stem] / disco_file.name
                shutil.move(str(disco_file), str(dest))
                moved += 1
        print(f"Moved {moved} disconnectome(s) into BIDS structure.")
    finally:
        shutil.rmtree(str(lesion_dir), ignore_errors=True)
        shutil.rmtree(str(disco_flat), ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
