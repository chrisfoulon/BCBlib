"""Tests for bcblib.imaging.manipulate."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.manipulate import extract_roi, merge_images, split_image, copy_geometry


def _make_nii(shape=(4, 5, 6), val=1.0, affine=None):
    data = np.full(shape, val, dtype=np.float64)
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


class TestExtractRoi:
    def test_spatial_crop(self):
        nii = _make_nii((10, 10, 10))
        roi = extract_roi(nii, x_min=2, x_size=3, y_min=1, y_size=4, z_min=0, z_size=5)
        assert roi.shape == (3, 4, 5)

    def test_default_full(self):
        nii = _make_nii((10, 10, 10))
        roi = extract_roi(nii)
        assert roi.shape == (10, 10, 10)

    def test_4d_time_crop(self):
        nii = _make_nii((5, 5, 5, 20))
        roi = extract_roi(nii, t_min=5, t_size=10)
        assert roi.shape == (5, 5, 5, 10)


class TestMergeImages:
    def test_merge_3d(self):
        imgs = [_make_nii((3, 3, 3), val=float(i)) for i in range(4)]
        merged = merge_images(imgs, axis=3)
        assert merged.shape == (3, 3, 3, 4)

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError):
            merge_images([])


class TestSplitImage:
    def test_round_trip(self):
        data = np.random.rand(3, 3, 3, 5)
        nii = nib.Nifti1Image(data, np.eye(4))
        vols = split_image(nii, axis=3)
        assert len(vols) == 5
        for i, v in enumerate(vols):
            np.testing.assert_allclose(v.get_fdata(), data[:, :, :, i])

    def test_split_3d_raises(self):
        nii = _make_nii((3, 3, 3))
        with pytest.raises(ValueError):
            split_image(nii, axis=3)


class TestCopyGeometry:
    def test_copies_affine(self):
        affine1 = np.diag([2, 2, 2, 1]).astype(float)
        affine2 = np.eye(4)
        src = _make_nii((3, 3, 3), affine=affine1)
        tgt = _make_nii((3, 3, 3), val=42.0, affine=affine2)

        result = copy_geometry(src, tgt)
        np.testing.assert_array_equal(result.affine, affine1)
        np.testing.assert_array_equal(result.get_fdata(), 42.0)
