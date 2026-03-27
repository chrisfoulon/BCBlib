"""Tests for bcblib.imaging.stats."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.stats import (
    centre_of_gravity,
    centre_of_gravity_distance,
    volume_count,
    laterality_ratio,
    reduce_axis,
    image_stats,
    percentile,
    robust_range,
    histogram,
)


def _make_nii(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(np.asarray(data, dtype=np.float64), affine)


class TestCentreOfGravity:
    def test_uniform_cube(self):
        data = np.ones((5, 5, 5))
        nii = _make_nii(data)
        cog = centre_of_gravity(nii)
        np.testing.assert_allclose(cog, (2.0, 2.0, 2.0))

    def test_empty_image(self):
        data = np.zeros((3, 3, 3))
        nii = _make_nii(data)
        cog = centre_of_gravity(nii)
        np.testing.assert_array_equal(cog, (0.0, 0.0, 0.0))

    def test_round(self):
        data = np.zeros((5, 5, 5))
        data[1, 2, 3] = 1
        nii = _make_nii(data)
        cog = centre_of_gravity(nii, round_coord=True)
        np.testing.assert_array_equal(cog, [1.0, 2.0, 3.0])


class TestCentreOfGravityDistance:
    def test_same_image(self):
        data = np.ones((5, 5, 5))
        nii = _make_nii(data)
        assert centre_of_gravity_distance(nii, nii) == 0.0

    def test_with_reference_coords(self):
        data = np.ones((5, 5, 5))
        nii = _make_nii(data)
        dist = centre_of_gravity_distance(nii, (0, 0, 0))
        assert dist > 0


class TestVolumeCount:
    def test_count(self):
        data = np.array([[[0, 1, 2], [3, 0, 0], [0, 0, 5]]], dtype=float)
        nii = _make_nii(data)
        assert volume_count(nii) == 4  # 1, 2, 3, 5

    def test_ratio(self):
        data = np.zeros((2, 2, 2))
        data[0, 0, 0] = 1
        nii = _make_nii(data)
        assert volume_count(nii, ratio=True) == pytest.approx(1 / 8)


class TestLateralityRatio:
    def test_left_dominant(self):
        data = np.zeros((10, 3, 3))
        data[7:, :, :] = 1  # right side in RAS
        nii = _make_nii(data)
        ratio = laterality_ratio(nii)
        # In RAS, higher x = right hemisphere
        assert ratio != 0.0


class TestReduceAxis:
    def test_mean_4d(self):
        data = np.random.rand(3, 3, 3, 5)
        nii = _make_nii(data)
        result = reduce_axis(nii, method="mean", axis=3)
        assert result.shape == (3, 3, 3)
        np.testing.assert_allclose(result.get_fdata(), data.mean(axis=3))


class TestImageStats:
    def test_basic(self):
        data = np.arange(27, dtype=float).reshape(3, 3, 3)
        nii = _make_nii(data)
        s = image_stats(nii)
        assert s["min"] == 0.0
        assert s["max"] == 26.0
        assert "mean" in s
        assert "std" in s
        assert s["total_voxels"] == 27

    def test_with_mask(self):
        data = np.ones((3, 3, 3)) * 10
        mask_data = np.zeros((3, 3, 3))
        mask_data[1, 1, 1] = 1
        nii = _make_nii(data)
        mask = _make_nii(mask_data)
        s = image_stats(nii, mask=mask)
        assert s["total_voxels"] == 1
        assert s["mean"] == 10.0


class TestPercentile:
    def test_median(self):
        data = np.arange(100, dtype=float).reshape(10, 10, 1)
        nii = _make_nii(data)
        p50 = percentile(nii, 50)
        assert p50 == pytest.approx(np.median(data), abs=1)


class TestRobustRange:
    def test_defaults(self):
        data = np.random.randn(10, 10, 10).astype(np.float64)
        nii = _make_nii(data)
        lo, hi = robust_range(nii)
        assert lo < hi


class TestHistogram:
    def test_shape(self):
        data = np.random.rand(5, 5, 5).astype(np.float64)
        nii = _make_nii(data)
        counts, edges = histogram(nii, bins=10)
        assert len(counts) == 10
        assert len(edges) == 11
