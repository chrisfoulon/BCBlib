"""Tests for bcblib.imaging.math."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.math import (
    binarize,
    threshold,
    dilate,
    erode,
    apply_mask,
    apply_inverse_mask,
    add,
    subtract,
    multiply,
    _dilate_array,
    _erode_array,
    _mask_in_array,
    _mask_out_array,
)


def _make_nii(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(np.asarray(data, dtype=np.float64), affine)


class TestBinarize:
    def test_with_threshold(self):
        data = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=float)
        nii = _make_nii(data)
        result = binarize(nii, thr=4)
        out = result.get_fdata()
        assert out[0, 1, 0] == 0  # value 3 < 4
        assert out[0, 1, 1] == 1  # value 4 >= 4
        assert out[0, 2, 2] == 1  # value 8 >= 4

    def test_already_binary(self):
        data = np.array([[[0, 1], [1, 0]]], dtype=float)
        nii = _make_nii(data)
        result = binarize(nii)
        assert result is nii  # returned unchanged

    def test_auto_threshold(self):
        data = np.array([[[0, 5, 10]]], dtype=float)
        nii = _make_nii(data)
        result = binarize(nii)
        out = result.get_fdata()
        assert out[0, 0, 0] == 0  # min value = background
        assert out[0, 0, 1] == 1
        assert out[0, 0, 2] == 1


class TestThreshold:
    def test_low(self):
        data = np.array([[[1, 5, 10]]], dtype=float)
        nii = _make_nii(data)
        result = threshold(nii, low=3)
        out = result.get_fdata()
        assert out[0, 0, 0] == 0  # 1 < 3
        assert out[0, 0, 1] == 5  # 5 >= 3

    def test_high(self):
        data = np.array([[[1, 5, 10]]], dtype=float)
        nii = _make_nii(data)
        result = threshold(nii, high=7)
        out = result.get_fdata()
        assert out[0, 0, 2] == 0  # 10 > 7
        assert out[0, 0, 1] == 5


class TestDilateErode:
    def test_dilate_expands(self):
        data = np.zeros((5, 5, 5))
        data[2, 2, 2] = 1
        nii = _make_nii(data)
        result = dilate(nii, connectivity=1)
        out = result.get_fdata()
        assert out[2, 2, 2] == 1
        assert out[2, 2, 3] == 1  # dilated neighbor
        assert np.sum(out) > 1

    def test_erode_shrinks(self):
        data = np.ones((5, 5, 5))
        nii = _make_nii(data)
        result = erode(nii, connectivity=1)
        out = result.get_fdata()
        # Corners should be eroded away
        assert out[0, 0, 0] == 0


class TestMasking:
    def test_apply_mask(self):
        data = np.ones((3, 3, 3)) * 10
        mask_data = np.zeros((3, 3, 3))
        mask_data[1, 1, 1] = 1
        result = apply_mask(_make_nii(data), _make_nii(mask_data))
        out = result.get_fdata()
        assert out[1, 1, 1] == 10
        assert out[0, 0, 0] == 0

    def test_apply_inverse_mask(self):
        data = np.ones((3, 3, 3)) * 10
        mask_data = np.zeros((3, 3, 3))
        mask_data[1, 1, 1] = 1
        result = apply_inverse_mask(_make_nii(data), _make_nii(mask_data))
        out = result.get_fdata()
        assert out[1, 1, 1] == 0
        assert out[0, 0, 0] == 10


class TestArithmetic:
    def test_add(self):
        a = _make_nii(np.ones((3, 3, 3)) * 2)
        b = _make_nii(np.ones((3, 3, 3)) * 3)
        result = add(a, b)
        np.testing.assert_array_equal(result.get_fdata(), 5.0)

    def test_subtract(self):
        a = _make_nii(np.ones((3, 3, 3)) * 5)
        b = _make_nii(np.ones((3, 3, 3)) * 3)
        result = subtract(a, b)
        np.testing.assert_array_equal(result.get_fdata(), 2.0)

    def test_multiply(self):
        a = _make_nii(np.ones((3, 3, 3)) * 4)
        b = _make_nii(np.ones((3, 3, 3)) * 3)
        result = multiply(a, b)
        np.testing.assert_array_equal(result.get_fdata(), 12.0)


class TestPrivateArrayHelpers:
    def test_dilate_array(self):
        arr = np.zeros((5, 5, 5), dtype=np.float64)
        arr[2, 2, 2] = 1
        result = _dilate_array(arr, connectivity=1)
        assert result.dtype == arr.dtype
        assert np.sum(result) > 1

    def test_mask_in_array(self):
        arr = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        out = _mask_in_array(arr, mask)
        assert out[1, 1, 1] == 1
        assert out[0, 0, 0] == 0

    def test_mask_out_array(self):
        arr = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        out = _mask_out_array(arr, mask)
        assert out[1, 1, 1] == 0
        assert out[0, 0, 0] == 1
