"""Tests for bcblib.imaging.info."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.info import (
    header_summary, header_dump, header_field,
    format_summary, format_summary_short, format_dump,
)


def _make_nii(shape=(4, 5, 6)):
    data = np.zeros(shape, dtype=np.float32)
    return nib.Nifti1Image(data, np.eye(4))


class TestHeaderField:
    def test_dim1(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii((4, 5, 6)), str(p))
        assert header_field(str(p), "dim1") == 4
        assert header_field(str(p), "dim2") == 5
        assert header_field(str(p), "dim3") == 6

    def test_pixdim(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        val = header_field(str(p), "pixdim1")
        assert isinstance(val, (int, float, np.floating))

    def test_unknown_field(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        with pytest.raises(KeyError):
            header_field(str(p), "nonexistent_field_xyz")


class TestHeaderSummary:
    def test_keys(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii((4, 5, 6)), str(p))
        info = header_summary(str(p))
        assert info["dim1"] == 4
        assert info["dim2"] == 5
        assert info["dim3"] == 6
        assert "orientation" in info
        assert "data_type" in info

    def test_extended_keys(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii((4, 5, 6)), str(p))
        info = header_summary(str(p))
        assert info["ndim"] == 3
        assert info["dimensions"] == (4, 5, 6)
        assert len(info["voxel_size"]) == 3
        assert isinstance(info["vox_units"], str)
        assert info["file_size"] > 0

    def test_format_grouped(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        info = header_summary(str(p))
        text = format_summary(info, styled=False)
        assert "File" in text
        assert "Dimensions" in text
        assert "Intensity" in text
        assert "x" in text  # dimension separator
        assert "mm" in text  # voxel size units

    def test_format_styled_has_ansi(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        info = header_summary(str(p))
        text = format_summary(info, styled=True)
        assert "\033[1m" in text  # bold escape

    def test_format_unstyled_no_ansi(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        info = header_summary(str(p))
        text = format_summary(info, styled=False)
        assert "\033[" not in text

    def test_format_short(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii((4, 5, 6)), str(p))
        info = header_summary(str(p))
        text = format_summary_short(info)
        assert "test.nii.gz" in text
        assert "float32" in text
        assert "[4, 5, 6]" in text


class TestHeaderDump:
    def test_dump_is_dict(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        dump = header_dump(str(p))
        assert isinstance(dump, dict)
        assert len(dump) > 10  # NIfTI-1 has many fields

    def test_format_dump(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        dump = header_dump(str(p))
        text = format_dump(dump)
        assert "\t" in text  # tab-aligned layout
        assert "=" not in text  # no longer uses ' = '
