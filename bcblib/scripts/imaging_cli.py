"""CLI entry points for bcblib.imaging (``bcb-info``, ``bcb-stats``, etc.)."""

import argparse
import json
import os
import sys


# -----------------------------------------------------------------------
# Terminal styling helpers
# -----------------------------------------------------------------------

def _use_style():
    """Return True when ANSI escape codes should be emitted."""
    if os.environ.get("NO_COLOR", ""):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _bold(text):
    """Wrap *text* in ANSI bold when the terminal supports it."""
    if _use_style():
        return f"\033[1m{text}\033[0m"
    return text


def _dim(text):
    """Wrap *text* in ANSI dim when the terminal supports it."""
    if _use_style():
        return f"\033[2m{text}\033[0m"
    return text


# -----------------------------------------------------------------------
# bcb-info  (fslinfo equivalent)
# -----------------------------------------------------------------------

def bcb_info():
    parser = argparse.ArgumentParser(
        prog="bcb-info",
        description="Print a concise summary of NIfTI header information (like fslinfo).",
    )
    parser.add_argument("image", help="Path to a NIfTI file")
    parser.add_argument("--short", action="store_true",
                        help="One-liner nib-ls style output")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI styling")
    args = parser.parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    from bcblib.imaging.info import header_summary, format_summary, format_summary_short

    info = header_summary(args.image)
    if args.short:
        print(format_summary_short(info))
    else:
        print(format_summary(info, styled=_use_style()))


# -----------------------------------------------------------------------
# bcb-header  (fslhd equivalent)
# -----------------------------------------------------------------------

def bcb_header():
    parser = argparse.ArgumentParser(
        prog="bcb-header",
        description="Dump all NIfTI header fields (like fslhd).",
    )
    parser.add_argument("image", help="Path to a NIfTI file")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    from bcblib.imaging.info import header_dump, format_dump

    dump = header_dump(args.image)
    if args.as_json:
        print(json.dumps(dump, indent=2, default=str))
    else:
        print(format_dump(dump))


# -----------------------------------------------------------------------
# bcb-stats  (fslstats equivalent)
# -----------------------------------------------------------------------

def _format_stats(stats, filename):
    """Format *stats* dict into grouped, human-friendly output."""
    from pathlib import Path
    basename = Path(filename).name

    lines = [
        _bold(f"Statistics for {basename}"),
        "",
        f"  {'min':<16s}  {stats['min']:.6f}",
        f"  {'max':<16s}  {stats['max']:.6f}",
        f"  {'mean':<16s}  {stats['mean']:.6f}",
        f"  {'std':<16s}  {stats['std']:.6f}",
        f"  {'median':<16s}  {stats['median']:.6f}",
        "",
        f"  {'robust range':<16s}  {stats['robust_min']:.2f} .. {stats['robust_max']:.2f}  {_dim('(2nd-98th percentile)')}",
        "",
        f"  {'nonzero':<16s}  {stats['nonzero_voxels']} voxels",
        f"  {'total':<16s}  {stats['total_voxels']} voxels",
        f"  {'volume':<16s}  {stats['volume_mm3']:.2f} mm\u00b3",
    ]
    return "\n".join(lines)


def bcb_stats():
    parser = argparse.ArgumentParser(
        prog="bcb-stats",
        description="Compute summary statistics for a NIfTI image (like fslstats).",
    )
    parser.add_argument("image", help="Path to a NIfTI file")
    parser.add_argument("-m", "--mask", default=None,
                        help="Optional mask image")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI styling")
    args = parser.parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    from bcblib.imaging.stats import image_stats

    stats = image_stats(args.image, mask=args.mask)
    if args.as_json:
        print(json.dumps(stats, indent=2))
    else:
        print(_format_stats(stats, args.image))


# -----------------------------------------------------------------------
# bcb-orient
# -----------------------------------------------------------------------

def bcb_orient():
    parser = argparse.ArgumentParser(
        prog="bcb-orient",
        description="Query or change image orientation.",
    )
    parser.add_argument("image", help="Path to a NIfTI file")
    parser.add_argument("--set", dest="target", default=None,
                        help="Target orientation code (e.g. RAS, LPI)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: print orientation)")
    args = parser.parse_args()

    from bcblib.imaging.orient import get_orientation, set_orientation
    from bcblib.imaging.io import load_nifti
    import nibabel as nib

    if args.target is None:
        ori = get_orientation(args.image)
        print("".join(ori))
    else:
        result = set_orientation(args.image, target=args.target)
        out = args.output or args.image
        nib.save(result, out)
        print(f"Saved with orientation {''.join(get_orientation(result))} -> {out}")


# -----------------------------------------------------------------------
# bcb-roi  (fslroi equivalent)
# -----------------------------------------------------------------------

def bcb_roi():
    parser = argparse.ArgumentParser(
        prog="bcb-roi",
        description="Extract a region of interest from a NIfTI image (like fslroi).",
    )
    parser.add_argument("image", help="Input NIfTI file")
    parser.add_argument("output", help="Output NIfTI file")
    parser.add_argument("xmin", type=int)
    parser.add_argument("xsize", type=int)
    parser.add_argument("ymin", type=int)
    parser.add_argument("ysize", type=int)
    parser.add_argument("zmin", type=int)
    parser.add_argument("zsize", type=int)
    parser.add_argument("tmin", type=int, nargs="?", default=0)
    parser.add_argument("tsize", type=int, nargs="?", default=-1)
    args = parser.parse_args()

    from bcblib.imaging.manipulate import extract_roi
    import nibabel as nib

    roi = extract_roi(
        args.image,
        x_min=args.xmin, x_size=args.xsize,
        y_min=args.ymin, y_size=args.ysize,
        z_min=args.zmin, z_size=args.zsize,
        t_min=args.tmin, t_size=args.tsize,
    )
    nib.save(roi, args.output)
    print(f"Saved ROI -> {args.output}")


# -----------------------------------------------------------------------
# bcb-merge  (fslmerge equivalent)
# -----------------------------------------------------------------------

def bcb_merge():
    parser = argparse.ArgumentParser(
        prog="bcb-merge",
        description="Merge NIfTI images along an axis (like fslmerge).",
    )
    parser.add_argument("output", help="Output NIfTI file")
    parser.add_argument("images", nargs="+", help="Input NIfTI files")
    parser.add_argument("-a", "--axis", type=int, default=3,
                        help="Axis to merge along (default: 3)")
    args = parser.parse_args()

    from bcblib.imaging.manipulate import merge_images
    import nibabel as nib

    merged = merge_images(args.images, axis=args.axis)
    nib.save(merged, args.output)
    print(f"Merged {len(args.images)} images -> {args.output}")


# -----------------------------------------------------------------------
# bcb-split  (fslsplit equivalent)
# -----------------------------------------------------------------------

def bcb_split():
    parser = argparse.ArgumentParser(
        prog="bcb-split",
        description="Split a 4-D NIfTI into individual volumes (like fslsplit).",
    )
    parser.add_argument("image", help="Input 4-D NIfTI file")
    parser.add_argument("output_prefix", help="Output prefix (e.g. vol_)")
    parser.add_argument("-a", "--axis", type=int, default=3,
                        help="Axis to split along (default: 3)")
    args = parser.parse_args()

    from bcblib.imaging.manipulate import split_image
    import nibabel as nib

    volumes = split_image(args.image, axis=args.axis)
    width = len(str(len(volumes) - 1))
    for i, vol in enumerate(volumes):
        out = f"{args.output_prefix}{str(i).zfill(width)}.nii.gz"
        nib.save(vol, out)
    print(f"Split into {len(volumes)} volumes with prefix '{args.output_prefix}'")


# -----------------------------------------------------------------------
# bcb-convert
# -----------------------------------------------------------------------

def bcb_convert():
    parser = argparse.ArgumentParser(
        prog="bcb-convert",
        description="Convert between .nii and .nii.gz formats.",
    )
    parser.add_argument("image", help="Input NIfTI file")
    parser.add_argument("output", help="Output NIfTI file (.nii or .nii.gz)")
    args = parser.parse_args()

    from bcblib.imaging.convert import convert_format

    out = convert_format(args.image, args.output)
    print(f"Converted -> {out}")
