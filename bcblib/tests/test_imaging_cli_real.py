"""Integration tests for bcblib.scripts.imaging_cli using real strokebrain images.

These tests are **read-only** — they never write to the strokebrain_test directory.
They are automatically **skipped** when the test data is not present (e.g. CI).

Image catalogue used:
  NATIVE_TRACEW : float32, shape 256x256x38, 0.94x0.94x4.5 mm (native DWI)
  NATIVE_ADC    : int16,   shape 256x256x38, 0.94x0.94x4.5 mm (native ADC)
  NATIVE_MASK   : uint8,   shape 256x256x38, 0.94x0.94x4.5 mm (lesion mask)
  MNI_ADC       : float32, shape 181x217x181, 1.0x1.0x1.0 mm (MNI-registered)
  SUB012_TRACEW : float32, shape varies (different subject native DWI)
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants (data-root guard)
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/chrisfoulon/neuro_data/strokebrain_test/vascprep_preprocessed")

NATIVE_TRACEW = DATA_ROOT / "BBS_sub-010/01_canonicalize/BBS_sub-010/BBS_sub-010_tracew.nii.gz"
NATIVE_ADC    = DATA_ROOT / "BBS_sub-010/01_canonicalize/BBS_sub-010/BBS_sub-010_adc.nii.gz"
NATIVE_MASK   = DATA_ROOT / "BBS_sub-010/01_canonicalize/BBS_sub-010/BBS_sub-010_tracew_lesion_mask_headerfix.nii.gz"
MNI_ADC       = DATA_ROOT / "BBS_sub-010/08_registration/BBS_sub-010_adc_registration.nii.gz"
SUB012_TRACEW = DATA_ROOT / "BBS_sub-012/01_canonicalize/BBS_sub-012/BBS_sub-012_tracew.nii.gz"

real_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="strokebrain test data not present",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mtime(path: Path) -> float:
    return path.stat().st_mtime


# ---------------------------------------------------------------------------
# bcb-info tests
# ---------------------------------------------------------------------------

@real_data
class TestBcbInfoReal:
    """bcb-info with real NIfTI images."""

    def test_single_image_renders_panel(self, capsys):
        """Single image: output should be wrapped in a rich Panel (box borders)."""
        sys.argv = ["bcb-info", str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        # Rich Panel uses box-drawing characters
        assert "╭" in out or "┌" in out or "BBS_sub-010_tracew.nii.gz" in out
        assert "float32" in out
        assert "256" in out  # spatial dim

    def test_single_image_contains_key_fields(self, capsys):
        """Verify specific expected values from the tracew header."""
        sys.argv = ["bcb-info", str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "float32" in out          # dtype
        assert "256 x 256 x 38" in out   # dimensions
        assert "0.94" in out             # voxel size
        assert "RAS" in out              # orientation

    def test_single_int16_image(self, capsys):
        """ADC image (int16) shows correct dtype."""
        sys.argv = ["bcb-info", str(NATIVE_ADC)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "int16" in out
        assert "256 x 256 x 38" in out

    def test_multi_three_images(self, capsys):
        """Three images from the same subject: all file names appear in output."""
        sys.argv = ["bcb-info", str(NATIVE_TRACEW), str(NATIVE_ADC), str(NATIVE_MASK)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "BBS_sub-010_tracew.nii.gz" in out
        assert "BBS_sub-010_adc.nii.gz" in out
        assert "BBS_sub-010_tracew_lesion_mask_headerfix.nii.gz" in out

    def test_multi_cross_subject(self, capsys):
        """Compare tracew from sub-010 (native) and sub-012 (native): both names present."""
        sys.argv = ["bcb-info", str(NATIVE_TRACEW), str(SUB012_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "BBS_sub-010_tracew.nii.gz" in out
        assert "BBS_sub-012_tracew.nii.gz" in out
        # Sub-012 native DWI has different spatial dimensions from sub-010
        assert "256 x 256 x 38" in out      # sub-010

    def test_multi_native_vs_mni(self, capsys):
        """Compare native (256x256x38) and MNI-space (181x217x181) images."""
        sys.argv = ["bcb-info", str(NATIVE_ADC), str(MNI_ADC)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "256 x 256 x 38" in out
        assert "181 x 217 x 181" in out

    def test_short_single(self, capsys):
        """--short produces a single one-liner for one image."""
        sys.argv = ["bcb-info", "--short", str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l.strip()]
        assert len(lines) == 1
        assert "BBS_sub-010_tracew.nii.gz" in lines[0]
        assert "float32" in lines[0]
        assert "[256, 256, 38]" in lines[0]
        assert "RAS" in lines[0]

    def test_short_multi_one_line_per_image(self, capsys):
        """--short with three images produces exactly three lines."""
        sys.argv = ["bcb-info", "--short",
                    str(NATIVE_TRACEW), str(NATIVE_ADC), str(NATIVE_MASK)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l.strip()]
        assert len(lines) == 3
        assert "BBS_sub-010_tracew.nii.gz" in lines[0]
        assert "BBS_sub-010_adc.nii.gz" in lines[1]
        assert "BBS_sub-010_tracew_lesion_mask_headerfix.nii.gz" in lines[2]

    def test_no_color_flag_suppresses_ansi(self, capsys, monkeypatch):
        """--no-color: output must contain no ANSI escape sequences."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        sys.argv = ["bcb-info", "--no-color", str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "\033[" not in out

    def test_does_not_modify_files(self):
        """Strict read-only check: mtime of all source images is unchanged."""
        mtimes_before = {p: _mtime(p) for p in
                         [NATIVE_TRACEW, NATIVE_ADC, NATIVE_MASK, MNI_ADC]}
        sys.argv = ["bcb-info", str(NATIVE_TRACEW), str(NATIVE_ADC),
                    str(NATIVE_MASK), str(MNI_ADC)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        for p, mtime in mtimes_before.items():
            assert _mtime(p) == mtime, f"File was unexpectedly modified: {p}"


# ---------------------------------------------------------------------------
# bcb-header tests
# ---------------------------------------------------------------------------

@real_data
class TestBcbHeaderReal:
    """bcb-header with real NIfTI images."""

    def test_single_dumps_all_fields(self, capsys):
        """Single image: every expected NIfTI-1 field should appear."""
        sys.argv = ["bcb-header", str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        for field in ("dim", "pixdim", "datatype", "bitpix", "vox_offset"):
            assert field in out, f"Missing expected header field: {field}"

    def test_single_json_is_valid_dict(self, capsys):
        """--json with one image returns a flat JSON object."""
        sys.argv = ["bcb-header", "--json", str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)
        assert "dim" in data
        assert "pixdim" in data
        # Dimensions: [3, 256, 256, 38, ...]
        assert data["dim"][1] == 256

    def test_compare_native_vs_mni_shows_diffs(self, capsys):
        """Comparison of native vs MNI-space images must flag dim and pixdim as differing."""
        sys.argv = ["bcb-header", str(NATIVE_ADC), str(MNI_ADC)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        # Both filenames (or their truncated versions) should appear
        assert "sub-010_adc" in out          # part of either truncated form
        assert "registration" in out
        # The comparison must note that something differs
        assert "*" in out                    # diff marker
        # The diff count message
        assert "differ" in out or "*" in out

    def test_compare_native_vs_mni_diff_count_nonzero(self, capsys):
        """The 'N fields differ' footer should report at least the dim/pixdim differences."""
        sys.argv = ["bcb-header", str(NATIVE_ADC), str(MNI_ADC)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        # Extract the numeric diff count from the footer line
        import re
        m = re.search(r"(\d+)\s+field", out)
        assert m is not None, "Expected 'N field(s) differ' footer not found"
        assert int(m.group(1)) >= 2   # at minimum: dim + pixdim differ

    def test_compare_same_image_twice_shows_no_diffs(self, capsys):
        """Comparing an image against itself should yield zero differing fields."""
        sys.argv = ["bcb-header", str(NATIVE_TRACEW), str(NATIVE_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        # The diff footer should NOT appear (no differences)
        assert "differ" not in out

    def test_compare_same_modality_cross_subject(self, capsys):
        """Two tracew images from different subjects: dim should differ (different FOV)."""
        sys.argv = ["bcb-header", str(NATIVE_TRACEW), str(SUB012_TRACEW)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "*" in out
        assert "differ" in out

    def test_sequential_flag_prints_headers_separately(self, capsys):
        """--sequential: each header block is labelled with its filename."""
        sys.argv = ["bcb-header", "--sequential",
                    str(NATIVE_TRACEW), str(NATIVE_ADC)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "BBS_sub-010_tracew.nii.gz" in out
        assert "BBS_sub-010_adc.nii.gz" in out
        # In sequential mode there is no comparison table, no diff marker
        assert "differ" not in out

    def test_json_multi_keyed_by_path(self, capsys):
        """--json with two images returns a dict keyed by the full input paths."""
        sys.argv = ["bcb-header", "--json", str(NATIVE_TRACEW), str(NATIVE_ADC)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)
        assert str(NATIVE_TRACEW) in data
        assert str(NATIVE_ADC) in data
        # Sanity: both images have the same dims
        assert data[str(NATIVE_TRACEW)]["dim"] == data[str(NATIVE_ADC)]["dim"]

    def test_does_not_modify_files(self):
        """Strict read-only check: no source file mtime changes after bcb-header."""
        paths = [NATIVE_TRACEW, NATIVE_ADC, MNI_ADC]
        mtimes_before = {p: _mtime(p) for p in paths}
        sys.argv = ["bcb-header", str(NATIVE_TRACEW), str(MNI_ADC)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        for p, mtime in mtimes_before.items():
            assert _mtime(p) == mtime, f"File was unexpectedly modified: {p}"
