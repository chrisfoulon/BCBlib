"""Tests for bcblib.imaging.convert."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.convert import convert_format, create_image


def _make_nii(shape=(3, 3, 3)):
    data = np.ones(shape, dtype=np.float32)
    return nib.Nifti1Image(data, np.eye(4))


class TestConvertFormat:
    def test_nii_to_nii_gz(self, tmp_path):
        p = tmp_path / "test.nii"
        nib.save(_make_nii(), str(p))
        out = convert_format(str(p), tmp_path / "test.nii.gz")
        assert out.exists()
        assert str(out).endswith(".nii.gz")

    def test_invalid_extension(self, tmp_path):
        p = tmp_path / "test.nii"
        nib.save(_make_nii(), str(p))
        with pytest.raises(ValueError):
            convert_format(str(p), tmp_path / "test.txt")


class TestCreateImage:
    def test_basic(self):
        img = create_image((4, 5, 6), fill=7.0)
        assert img.shape == (4, 5, 6)
        np.testing.assert_array_equal(img.get_fdata(), 7.0)

    def test_with_like(self, tmp_path):
        ref = _make_nii((10, 10, 10))
        p = tmp_path / "ref.nii.gz"
        nib.save(ref, str(p))
        img = create_image(shape=(10, 10, 10), like=str(p), fill=1.0)
        np.testing.assert_array_equal(img.affine, ref.affine)
        assert img.shape == (10, 10, 10)

    def test_default_affine(self):
        img = create_image((2, 2, 2))
        np.testing.assert_array_equal(img.affine, np.eye(4))
