"""Tests for bcblib.imaging.io."""

import numpy as np
import nibabel as nib
import pytest

from bcblib.imaging.io import is_nifti, load_nifti, resave_nifti, resave_nifti_list


def _make_nii(shape=(3, 3, 3), val=1.0):
    data = np.full(shape, val, dtype=np.float64)
    return nib.Nifti1Image(data, np.eye(4))


class TestIsNifti:
    def test_nii(self):
        assert is_nifti("brain.nii") is True

    def test_nii_gz(self):
        assert is_nifti("brain.nii.gz") is True

    def test_other(self):
        assert is_nifti("brain.txt") is False
        assert is_nifti("brain.nii.bz2") is False


class TestLoadNifti:
    def test_load_from_path(self, tmp_path):
        p = tmp_path / "test.nii.gz"
        nib.save(_make_nii(), str(p))
        result = load_nifti(str(p))
        assert isinstance(result, nib.Nifti1Image)

    def test_load_passthrough(self):
        nii = _make_nii()
        assert load_nifti(nii) is nii

    def test_load_invalid(self):
        with pytest.raises(ValueError):
            load_nifti("/nonexistent/path.nii.gz")

    def test_load_non_nifti(self):
        with pytest.raises(ValueError):
            load_nifti(42)


class TestResaveNifti:
    def test_resave_to_dir(self, tmp_path):
        p = tmp_path / "orig.nii.gz"
        nib.save(_make_nii(val=5.0), str(p))
        out = resave_nifti(str(p), output=tmp_path)
        assert out.exists()
        reloaded = nib.load(str(out))
        np.testing.assert_array_equal(reloaded.get_fdata(), 5.0)

    def test_resave_to_file(self, tmp_path):
        p = tmp_path / "orig.nii.gz"
        dest = tmp_path / "dest.nii.gz"
        nib.save(_make_nii(val=3.0), str(p))
        out = resave_nifti(str(p), output=dest)
        assert out == dest


class TestResaveNiftiList:
    def test_resave_list(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.nii.gz"
            nib.save(_make_nii(val=float(i)), str(p))
            paths.append(str(p))

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        resaved, failed = resave_nifti_list(paths, output_dir=out_dir)
        assert len(resaved) == 3
        assert len(failed) == 0
