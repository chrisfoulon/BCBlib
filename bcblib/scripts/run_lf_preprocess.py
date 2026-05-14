"""CLI: bcb-lf-preprocess — normalise lesions and compute disconnectomes."""

import argparse
import sys
from pathlib import Path


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
        help="Input BIDS directory containing sub-*/anat/*_label-lesion_mask.nii.gz",
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

    if not bids_dir.exists():
        print(f"ERROR: bids-dir does not exist: {bids_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        from bcblib.tools.lesion_features._bids import iter_bids_lesions
        subjects = list(iter_bids_lesions(bids_dir))
        print(f"Would preprocess {len(subjects)} lesion(s):")
        for sub_id, ses_id, path in subjects:
            ses_part = f" ses-{ses_id}" if ses_id else ""
            print(f"  sub-{sub_id}{ses_part}: {path}")
        return

    from bcblib.tools.lesion_features._pipeline import preprocess_batch
    from bcblib.tools.lesion_features._disco import find_bcbtoolkit, run_disco_batch

    print(f"Preprocessing lesions → {output_dir}")
    results = preprocess_batch(bids_dir, output_dir, force=force)
    print(f"Preprocessed {len(results)} subject(s).")

    try:
        kit = find_bcbtoolkit(args.bcbtoolkit)
    except FileNotFoundError as e:
        print(f"WARNING: {e}\nSkipping disconnectome computation.", file=sys.stderr)
        return

    # Collect lesion dir(s) and run disco per session/subject
    from bcblib.tools.lesion_features._bids import iter_bids_lesions
    import shutil

    # Flatten all normalised lesions into a tmp dir for run_disco.sh (folder mode)
    # keeping a mapping so we can move disconnectomes back into BIDS structure.
    lesion_dir = output_dir / "_tmp_lesions_for_disco"
    lesion_dir.mkdir(parents=True, exist_ok=True)
    disco_flat = output_dir / "_tmp_disco_flat"
    sub_map = {}  # filename_stem → (sub_id, ses_id, anat_dir)
    try:
        for sub_id, ses_id, lesion_path in iter_bids_lesions(output_dir):
            shutil.copy2(str(lesion_path), str(lesion_dir / lesion_path.name))
            stem = lesion_path.name.replace("_label-lesion_mask.nii.gz", "_desc-disconnectome")
            sub_map[stem] = (sub_id, ses_id, lesion_path.parent)

        n = len(list(lesion_dir.glob("*.nii.gz")))
        print(f"Running disconnectome computation for {n} subjects...")
        run_disco_batch(
            lesion_dir, disco_flat, kit,
            ncores=args.ncores, tracks_dir=args.tracks_dir,
            tmpdir=args.tmpdir,
        )

        # Move disconnectomes into the per-subject BIDS anat directories
        moved = 0
        for disco_file in sorted(disco_flat.glob("*_desc-disconnectome.nii.gz")):
            stem = disco_file.name.replace(".nii.gz", "")
            if stem in sub_map:
                _, _, anat_dir = sub_map[stem]
                dest = anat_dir / disco_file.name
                shutil.move(str(disco_file), str(dest))
                moved += 1
        print(f"Moved {moved} disconnectome(s) into BIDS structure.")
    finally:
        shutil.rmtree(str(lesion_dir), ignore_errors=True)
        shutil.rmtree(str(disco_flat), ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
