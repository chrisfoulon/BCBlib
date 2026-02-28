"""CLI entry points for bcblib.imaging (``bcb-info``, ``bcb-stats``, etc.)."""

import argparse
import json
import os
import sys
from pathlib import Path


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
# Multi-image rendering helpers
# -----------------------------------------------------------------------

def _render_info_multi(infos, styled):
    """Print multiple header summaries.

    When *rich* is available, renders side-by-side panels sizing to the
    terminal width.  Falls back to sequentially separated blocks otherwise.
    """
    from bcblib.imaging.info import format_summary, format_summary_short

    try:
        from rich.console import Console
        from rich.columns import Columns
        from rich.panel import Panel

        panels = []
        for info in infos:
            basename = Path(info.get("filename", "<in-memory>")).name
            body = format_summary(info, styled=False)
            panels.append(
                Panel(body, title=f"[bold cyan]{basename}[/bold cyan]", expand=True)
            )
        Console().print(Columns(panels, equal=True))

    except ImportError:
        for i, info in enumerate(infos):
            if i > 0:
                print()
            basename = Path(info.get("filename", "<in-memory>")).name
            bar = "─" * max(0, 52 - len(basename))
            print(f"──  {basename}  {bar}")
            print(format_summary(info, styled=styled))


def _render_header_compare(dumps, filenames):
    """Print a field-by-field comparison table for multiple images.

    Uses *rich* when available (differences highlighted in yellow).
    Falls back to a plain fixed-width text table otherwise.
    """
    names = [Path(f).name for f in filenames]
    all_fields = list(dumps[0].keys())

    def _differs(field):
        return len({str(d.get(field, "")) for d in dumps}) > 1

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        table = Table(
            title="NIfTI Header Comparison",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            highlight=False,
        )
        table.add_column("Field", style="bold", no_wrap=True, min_width=22)
        for name in names:
            table.add_column(name, overflow="fold", min_width=14)

        diff_count = 0
        for field in all_fields:
            values = [str(d.get(field, "")) for d in dumps]
            differs = len(set(values)) > 1
            row_style = "yellow" if differs else ""
            marker = " [dim]*[/dim]" if differs else ""
            table.add_row(field + marker, *values, style=row_style)
            if differs:
                diff_count += 1

        console = Console()
        console.print(table)
        if diff_count:
            console.print(
                f"[dim]{diff_count} field(s) marked [yellow]*[/yellow] differ between images.[/dim]"
            )

    except ImportError:
        # Plain-text fallback
        col0_w = max(len("Field"), max(len(f) for f in all_fields)) + 2
        col_ws = [
            max(len(n), max(len(str(d.get(f, ""))) for f in all_fields))
            for n, d in zip(names, dumps)
        ]
        sep = "  "
        header = f"{'Field':<{col0_w}}" + sep + sep.join(
            f"{n:<{w}}" for n, w in zip(names, col_ws)
        )
        print(header)
        print("-" * len(header))
        for field in all_fields:
            vals = [str(d.get(field, "")) for d in dumps]
            marker = " *" if _differs(field) else "  "
            row = f"{(field + marker):<{col0_w}}" + sep + sep.join(
                f"{v:<{w}}" for v, w in zip(vals, col_ws)
            )
            print(row)


# -----------------------------------------------------------------------
# bcb-info  (fslinfo equivalent)
# -----------------------------------------------------------------------

def bcb_info():
    parser = argparse.ArgumentParser(
        prog="bcb-info",
        description=(
            "Print a concise summary of NIfTI header information (like fslinfo). "
            "Accepts one or more images; with multiple images the summaries are "
            "displayed side-by-side when the terminal is wide enough."
        ),
    )
    parser.add_argument("images", nargs="+", help="Path(s) to NIfTI file(s)")
    parser.add_argument("--short", action="store_true",
                        help="One-liner nib-ls style output (one line per image)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI styling")
    args = parser.parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    from bcblib.imaging.info import header_summary, format_summary, format_summary_short

    infos = [header_summary(img) for img in args.images]

    if args.short:
        for info in infos:
            print(format_summary_short(info))
        return

    if len(infos) == 1:
        print(format_summary(infos[0], styled=_use_style()))
    else:
        _render_info_multi(infos, styled=_use_style())


# -----------------------------------------------------------------------
# bcb-header  (fslhd equivalent)
# -----------------------------------------------------------------------

def bcb_header():
    parser = argparse.ArgumentParser(
        prog="bcb-header",
        description=(
            "Dump all NIfTI header fields (like fslhd). "
            "With multiple images, shows a side-by-side comparison table; "
            "use --sequential to print each header separately instead."
        ),
    )
    parser.add_argument("images", nargs="+", help="Path(s) to NIfTI file(s)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON (array for multiple images)")
    parser.add_argument("--sequential", action="store_true",
                        help="Print each header separately even when multiple images are given")
    args = parser.parse_args()

    from bcblib.imaging.info import header_dump, format_dump

    dumps = [header_dump(img) for img in args.images]

    if args.as_json:
        if len(args.images) == 1:
            print(json.dumps(dumps[0], indent=2, default=str))
        else:
            result = {img: dump for img, dump in zip(args.images, dumps)}
            print(json.dumps(result, indent=2, default=str))
        return

    if len(dumps) == 1 or args.sequential:
        for i, (img, dump) in enumerate(zip(args.images, dumps)):
            if i > 0:
                print()
            if len(dumps) > 1:
                print(f"── {Path(img).name} " + "─" * max(0, 52 - len(Path(img).name)))
            print(format_dump(dump))
    else:
        _render_header_compare(dumps, args.images)


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
