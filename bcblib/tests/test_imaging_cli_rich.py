"""Tests for the rich-formatted multi-image features of bcblib.scripts.imaging_cli.

All images are synthetic (numpy + nibabel) built in ``tmp_path`` — no external
data required.  The fixtures deliberately mirror realistic neuroimaging scenarios:

  nii_float32  : float32, shape (12, 14, 10),  1 mm isotropic  — "T1/tracew"
  nii_int16    : int16,   shape (12, 14, 10),  1 mm isotropic  — "ADC"
  nii_mask     : uint8,   shape (12, 14, 10),  1 mm isotropic  — "lesion mask"
  nii_mni      : float32, shape (9, 11, 9),    2 mm isotropic  — "MNI-space"
  nii_other    : float32, shape (8, 9, 7),     0.94x0.94x4.5mm — "other subject native"
"""

import json
import sys

import nibabel as nib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _save(tmp_path, name, data, affine):
    p = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine), str(p))
    return p


@pytest.fixture
def nii_float32(tmp_path):
    data = np.random.rand(12, 14, 10).astype(np.float32)
    return _save(tmp_path, "tracew.nii.gz", data, np.eye(4))


@pytest.fixture
def nii_int16(tmp_path):
    data = (np.random.rand(12, 14, 10) * 1000).astype(np.int16)
    return _save(tmp_path, "adc.nii.gz", data, np.eye(4))


@pytest.fixture
def nii_mask(tmp_path):
    data = np.zeros((12, 14, 10), dtype=np.uint8)
    data[4:8, 5:9, 3:7] = 1
    return _save(tmp_path, "lesion_mask.nii.gz", data, np.eye(4))


@pytest.fixture
def nii_mni(tmp_path):
    """Different shape and 2 mm voxels — simulates MNI-registered image."""
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    data = np.random.rand(9, 11, 9).astype(np.float32)
    return _save(tmp_path, "mni_registration.nii.gz", data, aff)


@pytest.fixture
def nii_other(tmp_path):
    """Anisotropic voxels and different shape — simulates another subject's native DWI."""
    aff = np.diag([0.94, 0.94, 4.5, 1.0])
    data = np.random.rand(8, 9, 7).astype(np.float32)
    return _save(tmp_path, "other_subject_tracew.nii.gz", data, aff)


# ---------------------------------------------------------------------------
# bcb-info: single-image Panel rendering
# ---------------------------------------------------------------------------

class TestBcbInfoSingleRich:

    def test_panel_border_chars_present(self, nii_float32, capsys):
        """Rich Panel outputs box-drawing characters."""
        sys.argv = ["bcb-info", str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        # Rich uses rounded box borders: ╭ / ╰  (or at least some box chars)
        assert any(c in out for c in ("╭", "┌", "╔", "─"))

    def test_single_float32_content(self, nii_float32, capsys):
        """Key fields appear in a float32 image summary."""
        sys.argv = ["bcb-info", str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "float32" in out
        assert "12 x 14 x 10" in out
        assert "tracew.nii.gz" in out
        assert "RAS" in out           # identity affine → RAS

    def test_single_int16_content(self, nii_int16, capsys):
        sys.argv = ["bcb-info", str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "int16" in out
        assert "12 x 14 x 10" in out
        assert "adc.nii.gz" in out

    def test_single_uint8_mask(self, nii_mask, capsys):
        sys.argv = ["bcb-info", str(nii_mask)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "uint8" in out
        assert "lesion_mask.nii.gz" in out

    def test_mni_image_shows_2mm_voxels(self, nii_mni, capsys):
        sys.argv = ["bcb-info", str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "9 x 11 x 9" in out
        assert "2.00" in out           # 2 mm isotropic

    def test_no_color_no_ansi(self, nii_float32, capsys, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        sys.argv = ["bcb-info", "--no-color", str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        assert "\033[" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# bcb-info: multi-image side-by-side
# ---------------------------------------------------------------------------

class TestBcbInfoMultiRich:

    def test_two_same_modality_both_appear(self, nii_float32, nii_other, capsys):
        """Two files: both filenames visible."""
        sys.argv = ["bcb-info", str(nii_float32), str(nii_other)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "tracew.nii.gz" in out
        assert "other_subject_tracew.nii.gz" in out

    def test_three_images_all_names_present(self, nii_float32, nii_int16, nii_mask, capsys):
        """Three images: all three filenames present."""
        sys.argv = ["bcb-info",
                    str(nii_float32), str(nii_int16), str(nii_mask)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        for name in ("tracew.nii.gz", "adc.nii.gz", "lesion_mask.nii.gz"):
            assert name in out

    def test_native_vs_mni_both_shapes_visible(self, nii_float32, nii_mni, capsys):
        """Native (12x14x10) and MNI (9x11x9) dimensions both appear in output."""
        sys.argv = ["bcb-info", str(nii_float32), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "12 x 14 x 10" in out
        assert "9 x 11 x 9" in out

    def test_multi_different_dtypes_visible(self, nii_float32, nii_int16, nii_mask, capsys):
        """All three dtypes (float32, int16, uint8) visible in a single call."""
        sys.argv = ["bcb-info",
                    str(nii_float32), str(nii_int16), str(nii_mask)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        out = capsys.readouterr().out
        assert "float32" in out
        assert "int16" in out
        assert "uint8" in out

    def test_short_single_line_per_image(self, nii_float32, nii_int16, nii_mask, capsys):
        """--short with 3 images → exactly 3 non-empty lines."""
        sys.argv = ["bcb-info", "--short",
                    str(nii_float32), str(nii_int16), str(nii_mask)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        lines = [l for l in capsys.readouterr().out.strip().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_short_line_order_matches_arg_order(self, nii_float32, nii_int16, capsys):
        """--short: first line corresponds to first argument."""
        sys.argv = ["bcb-info", "--short", str(nii_float32), str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        lines = [l for l in capsys.readouterr().out.strip().splitlines() if l.strip()]
        assert "tracew.nii.gz" in lines[0]
        assert "adc.nii.gz" in lines[1]

    def test_short_line_contains_dtype_dims_orientation(self, nii_float32, capsys):
        """--short one-liner contains dtype, bracketed dims and orientation code."""
        sys.argv = ["bcb-info", "--short", str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        line = capsys.readouterr().out.strip()
        assert "float32" in line
        assert "[12, 14, 10]" in line
        assert "RAS" in line

    def test_does_not_modify_files(self, nii_float32, nii_int16, nii_mask):
        """All source files are strictly read-only — mtime must not change."""
        paths = [nii_float32, nii_int16, nii_mask]
        mtimes = {p: p.stat().st_mtime for p in paths}
        sys.argv = ["bcb-info", str(nii_float32), str(nii_int16), str(nii_mask)]
        from bcblib.scripts.imaging_cli import bcb_info
        bcb_info()
        for p, mtime in mtimes.items():
            assert p.stat().st_mtime == mtime, f"File unexpectedly modified: {p}"


# ---------------------------------------------------------------------------
# bcb-header: single image
# ---------------------------------------------------------------------------

class TestBcbHeaderSingle:

    def test_all_standard_fields_present(self, nii_float32, capsys):
        """Standard NIfTI-1 fields must all appear in the dump."""
        sys.argv = ["bcb-header", str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        for field in ("dim", "pixdim", "datatype", "bitpix", "vox_offset"):
            assert field in out, f"Missing header field: {field!r}"

    def test_json_is_valid_flat_dict(self, nii_float32, capsys):
        """--json returns a flat JSON object with the expected dim structure."""
        sys.argv = ["bcb-header", "--json", str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)
        assert "dim" in data and "pixdim" in data
        # dim[0] is ndim (3), dim[1] is x-size (12)
        assert data["dim"][0] == 3
        assert data["dim"][1] == 12

    def test_json_dtype_matches(self, nii_int16, capsys):
        """datatype code for int16 is 4."""
        sys.argv = ["bcb-header", "--json", str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        assert data["datatype"] == 4    # NIfTI datatype code for int16


# ---------------------------------------------------------------------------
# bcb-header: comparison table
# ---------------------------------------------------------------------------

class TestBcbHeaderCompare:

    def test_same_image_twice_no_diffs(self, nii_float32, capsys):
        """Comparing an image with itself must yield zero differing fields."""
        sys.argv = ["bcb-header", str(nii_float32), str(nii_float32)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "differ" not in out

    def test_different_shapes_flags_diffs(self, nii_float32, nii_mni, capsys):
        """Native (12x14x10) vs MNI (9x11x9): dim and pixdim should be flagged."""
        sys.argv = ["bcb-header", str(nii_float32), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "*" in out
        assert "differ" in out

    def test_diff_count_at_least_two(self, nii_float32, nii_mni, capsys):
        """dim + pixdim both differ between native and MNI images."""
        import re
        sys.argv = ["bcb-header", str(nii_float32), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        m = re.search(r"(\d+)\s+field", out)
        assert m is not None, "Expected 'N field(s) differ' footer not found"
        assert int(m.group(1)) >= 2

    def test_different_dtypes_flags_datatype(self, nii_float32, nii_int16, capsys):
        """float32 vs int16: datatype field must be flagged as differing."""
        sys.argv = ["bcb-header", str(nii_float32), str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "*" in out
        assert "differ" in out

    def test_both_filenames_in_output(self, nii_float32, nii_mni, capsys):
        """Both (possibly truncated) file names must appear as column headers."""
        sys.argv = ["bcb-header", str(nii_float32), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        # Filenames may be truncated on the left; check the invariant suffix
        assert "tracew.nii.gz" in out
        assert "mni_registration.nii.gz" in out

    def test_three_image_comparison(self, nii_float32, nii_int16, nii_mni, capsys):
        """Three-way comparison runs without error and all names appear.

        Short-enough names are shown in full; long names may be left-truncated
        with '…' by ``_short_name()``.  We check each name via a suffix that
        is always preserved regardless of truncation.
        """
        sys.argv = ["bcb-header",
                    str(nii_float32), str(nii_int16), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "tracew.nii.gz" in out         # short — shown in full
        assert "adc.nii.gz" in out            # short — shown in full
        # "mni_registration.nii.gz" may be left-truncated to "…tration.nii.gz"
        assert "tration.nii.gz" in out

    def test_sequential_flag_no_comparison_table(self, nii_float32, nii_mni, capsys):
        """--sequential prints each header separately; no diff markers."""
        sys.argv = ["bcb-header", "--sequential",
                    str(nii_float32), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        out = capsys.readouterr().out
        assert "tracew.nii.gz" in out
        assert "mni_registration.nii.gz" in out
        assert "differ" not in out     # no comparison footer in sequential mode

    def test_json_multi_keyed_by_input_paths(self, nii_float32, nii_int16, capsys):
        """--json with two images → dict keyed by the full input path strings."""
        sys.argv = ["bcb-header", "--json", str(nii_float32), str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        assert str(nii_float32) in data
        assert str(nii_int16) in data

    def test_json_multi_values_differ_where_expected(self, nii_float32, nii_int16, capsys):
        """In the per-path JSON, datatype codes differ between float32 and int16."""
        sys.argv = ["bcb-header", "--json", str(nii_float32), str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        dt_f32 = data[str(nii_float32)]["datatype"]
        dt_i16 = data[str(nii_int16)]["datatype"]
        assert dt_f32 != dt_i16   # 16 (float32) vs 4 (int16)

    def test_json_same_dims_for_same_shape_images(self, nii_float32, nii_int16, capsys):
        """float32 and int16 share the same shape → dim array must match in JSON."""
        sys.argv = ["bcb-header", "--json", str(nii_float32), str(nii_int16)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        data = json.loads(capsys.readouterr().out)
        assert data[str(nii_float32)]["dim"] == data[str(nii_int16)]["dim"]

    def test_does_not_modify_files(self, nii_float32, nii_mni):
        """Strict read-only: mtime of every source image must be unchanged."""
        paths = [nii_float32, nii_mni]
        mtimes = {p: p.stat().st_mtime for p in paths}
        sys.argv = ["bcb-header", str(nii_float32), str(nii_mni)]
        from bcblib.scripts.imaging_cli import bcb_header
        bcb_header()
        for p, mtime in mtimes.items():
            assert p.stat().st_mtime == mtime, f"File unexpectedly modified: {p}"
