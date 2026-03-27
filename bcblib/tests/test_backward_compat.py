"""Test backward compatibility: old import paths must still work (with deprecation warnings)."""

import warnings
import numpy as np
import nibabel as nib
import pytest


def _make_nii(tmp_path, name="compat.nii.gz", shape=(3, 3, 3)):
    p = tmp_path / name
    data = np.ones(shape, dtype=np.float64)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    return str(p)


class TestNiftiUtilsShims:
    """All moved functions from nifti_utils.py should emit DeprecationWarning."""

    def test_is_nifti(self):
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.io.is_nifti"):
            from bcblib.tools.nifti_utils import is_nifti
            assert is_nifti("brain.nii") is True

    def test_load_nifti(self, tmp_path):
        p = _make_nii(tmp_path)
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.io.load_nifti"):
            from bcblib.tools.nifti_utils import load_nifti
            result = load_nifti(p)
            assert isinstance(result, nib.Nifti1Image)

    def test_resave_nifti(self, tmp_path):
        p = _make_nii(tmp_path)
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.io.resave_nifti"):
            from bcblib.tools.nifti_utils import resave_nifti
            resave_nifti(p, output=tmp_path)

    def test_get_nifti_orientation(self, tmp_path):
        p = _make_nii(tmp_path)
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.orient.get_orientation"):
            from bcblib.tools.nifti_utils import get_nifti_orientation
            ori = get_nifti_orientation(p)
            assert len(ori) == 3

    def test_get_volume(self, tmp_path):
        p = _make_nii(tmp_path)
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.stats.volume_count"):
            from bcblib.tools.nifti_utils import get_volume
            v = get_volume(p)
            assert v == 27  # 3x3x3 all ones

    def test_binarize_nii(self, tmp_path):
        p = _make_nii(tmp_path)
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.math.binarize"):
            from bcblib.tools.nifti_utils import binarize_nii
            result = binarize_nii(p)
            assert isinstance(result, nib.Nifti1Image)

    def test_laterality_ratio(self, tmp_path):
        p = _make_nii(tmp_path)
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.stats.laterality_ratio"):
            from bcblib.tools.nifti_utils import laterality_ratio
            r = laterality_ratio(p)
            assert isinstance(r, float)


class TestNiiStatsShim:
    def test_simple_stats(self, tmp_path):
        p = tmp_path / "4d.nii.gz"
        data = np.random.rand(3, 3, 3, 5).astype(np.float64)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))

        with pytest.warns(DeprecationWarning, match="bcblib.imaging.stats.reduce_axis"):
            from bcblib.tools.nii_stats import simple_stats
            result = simple_stats(str(p), method="mean", axis=3)
            assert result.shape == (3, 3, 3)


class TestImagesUtilsShims:
    def test_dilate_mask(self):
        arr = np.zeros((5, 5, 5), dtype=np.float64)
        arr[2, 2, 2] = 1
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.math.dilate"):
            from bcblib.tools.images_utils import dilate_mask
            result = dilate_mask(arr, connectivity=1)
            assert np.sum(result) > 1

    def test_mask_in_array(self):
        arr = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.math.apply_mask"):
            from bcblib.tools.images_utils import mask_in_array
            result = mask_in_array(arr, mask)
            assert result[0, 0, 0] == 0
            assert result[1, 1, 1] == 1

    def test_mask_out_array(self):
        arr = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        with pytest.warns(DeprecationWarning, match="bcblib.imaging.math.apply_inverse_mask"):
            from bcblib.tools.images_utils import mask_out_array
            result = mask_out_array(arr, mask)
            assert result[1, 1, 1] == 0
            assert result[0, 0, 0] == 1


class TestMiscSplitRegression:
    """misc.py split functions must remain importable and produce correct output."""

    def _make_info_dict(self):
        """Build minimal synthetic info_dict matching create_balanced_split's format."""
        rng = np.random.default_rng(0)
        keys = [f"subj_{i:02d}" for i in range(30)]
        clusters = ["__AB", "__CD", "__EF"]  # bilat_offset=2 strips "xx" prefix
        info_dict = {
            k: {
                'lesion_cluster': clusters[i % len(clusters)],
                'volume': float(rng.integers(100, 5000)),
            }
            for i, k in enumerate(keys)
        }
        return keys, info_dict

    def test_import_still_works(self):
        from bcblib.tools.misc import permutation_balanced_splits  # noqa: F401

    def test_output_format(self):
        from bcblib.tools.misc import permutation_balanced_splits
        keys, info_dict = self._make_info_dict()
        result = permutation_balanced_splits(keys, info_dict, num_permutations=10)
        assert isinstance(result, list)
        assert len(result) == 5
        for split in result:
            assert isinstance(split, dict)


class TestNoCircularImports:
    """Verify no import cycle between imaging.* and tools.*."""

    def test_import_imaging_package(self):
        import importlib
        mod = importlib.import_module("bcblib.imaging")
        assert hasattr(mod, "load_nifti")

    def test_import_tools_after_imaging(self):
        import importlib
        importlib.import_module("bcblib.imaging")
        # This should not raise ImportError
        mod = importlib.import_module("bcblib.tools.nifti_utils")
        assert hasattr(mod, "load_nifti")

    def test_import_images_utils_after_imaging(self):
        import importlib
        importlib.import_module("bcblib.imaging")
        mod = importlib.import_module("bcblib.tools.images_utils")
        assert hasattr(mod, "dilate_mask")
