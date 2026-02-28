"""Tests for bcblib.imaging.orient."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.orient import get_orientation, reorient_to_standard, set_orientation, swap_dimensions


def _make_nii(shape=(4, 5, 6)):
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return nib.Nifti1Image(data, np.eye(4))


def _make_saved_nii(tmp_path, shape=(4, 5, 6)):
    nii = _make_nii(shape)
    p = tmp_path / "test.nii.gz"
    nib.save(nii, str(p))
    return nib.load(str(p))


class TestGetOrientation:
    def test_identity_affine(self):
        nii = _make_nii()
        assert get_orientation(nii) == ("R", "A", "S")

    def test_from_path(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        assert get_orientation(str(p)) == ("R", "A", "S")


class TestReorientToStandard:
    def test_already_canonical(self, tmp_path):
        nii = _make_saved_nii(tmp_path)
        result = reorient_to_standard(nii)
        assert isinstance(result, nib.Nifti1Image)

    def test_no_filename_raises(self):
        nii = _make_nii()
        with pytest.raises(ValueError):
            reorient_to_standard(nii)


class TestSetOrientation:
    def test_same_orientation(self):
        nii = _make_nii()
        result = set_orientation(nii, "RAS")
        assert get_orientation(result) == ("R", "A", "S")

    def test_flip_to_las(self):
        nii = _make_nii()
        result = set_orientation(nii, "LAS")
        assert get_orientation(result) == ("L", "A", "S")


class TestSwapDimensions:
    def test_swap_yz(self):
        nii = _make_nii((4, 5, 6))
        result = swap_dimensions(nii, ("x", "z", "y"))
        ori = get_orientation(result)
        # x stays the same, y and z swap
        assert ori[0] == "R"
        assert ori[1] == "S"  # was z
        assert ori[2] == "A"  # was y

    def test_negate_x(self):
        nii = _make_nii()
        result = swap_dimensions(nii, ("-x", "y", "z"))
        ori = get_orientation(result)
        assert ori[0] == "L"  # flipped from R
