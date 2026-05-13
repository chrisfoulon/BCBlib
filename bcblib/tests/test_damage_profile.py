"""Tests for bcblib.tools.damage_profile."""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(data: np.ndarray, affine=None) -> nib.Nifti1Image:
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data.astype(np.float32), affine)


def _save_nifti(path, data: np.ndarray, affine=None):
    nib.save(_make_nifti(data, affine), str(path))


# ---------------------------------------------------------------------------
# T2 — Atlas loading: directory format
# ---------------------------------------------------------------------------

class TestAtlasLoading:

    def test_directory_loads_all_files(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        for i in range(3):
            d = np.zeros((5, 5, 5), dtype=np.float32)
            d[i, i, i] = 1.0
            _save_nifti(tmp_path / f"region_{i:02d}.nii.gz", d)

        spec = AtlasSpec(source=str(tmp_path), name="test")
        result = load_atlas(spec)
        assert len(result) == 3

    def test_directory_threshold_applied(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        d = np.full((5, 5, 5), 0.3, dtype=np.float32)
        _save_nifti(tmp_path / "r.nii.gz", d)

        spec = AtlasSpec(source=str(tmp_path), name="test", threshold=0.5)
        result = load_atlas(spec)
        # All values 0.3 are below threshold 0.5 → all zeroed → region excluded
        assert len(result) == 0

    def test_directory_empty_regions_excluded(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        good = np.zeros((5, 5, 5), dtype=np.float32)
        good[2, 2, 2] = 1.0
        empty = np.zeros((5, 5, 5), dtype=np.float32)
        _save_nifti(tmp_path / "good.nii.gz", good)
        _save_nifti(tmp_path / "empty.nii.gz", empty)

        spec = AtlasSpec(source=str(tmp_path), name="test")
        result = load_atlas(spec)
        assert list(result.keys()) == ["good"]

    # T3 — 4D NIfTI

    def test_4d_nifti_loads_correct_n_regions(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5, 4), dtype=np.float32)
        for i in range(4):
            data[i, i, i, i] = 1.0
        f = tmp_path / "atlas4d.nii.gz"
        _save_nifti(f, data)

        spec = AtlasSpec(source=str(f), name="test")
        result = load_atlas(spec)
        assert len(result) == 4

    def test_4d_nifti_names_from_label_file(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5, 2), dtype=np.float32)
        data[0, 0, 0, 0] = 1.0
        data[1, 1, 1, 1] = 1.0
        f = tmp_path / "atlas4d.nii.gz"
        _save_nifti(f, data)
        label_f = tmp_path / "labels.txt"
        label_f.write_text("LeftHippocampus\nRightHippocampus\n")

        spec = AtlasSpec(source=str(f), name="test", label_file=str(label_f))
        result = load_atlas(spec)
        assert set(result.keys()) == {"LeftHippocampus", "RightHippocampus"}

    def test_4d_nifti_names_auto_numbered_without_label_file(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5, 2), dtype=np.float32)
        data[0, 0, 0, 0] = 1.0
        data[1, 1, 1, 1] = 1.0
        f = tmp_path / "atlas4d.nii.gz"
        _save_nifti(f, data)

        spec = AtlasSpec(source=str(f), name="test")
        result = load_atlas(spec)
        assert "region_0001" in result
        assert "region_0002" in result

    # T4 — Label NIfTI

    def test_label_nifti_correct_n_regions(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[0, 0, 0] = 1
        data[1, 1, 1] = 2
        data[2, 2, 2] = 3
        f = tmp_path / "labels.nii.gz"
        _save_nifti(f, data)

        spec = AtlasSpec(source=str(f), name="test")
        result = load_atlas(spec)
        assert len(result) == 3

    def test_label_nifti_names_from_label_file(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[0, 0, 0] = 1
        data[1, 1, 1] = 2
        f = tmp_path / "labels.nii.gz"
        _save_nifti(f, data)
        label_f = tmp_path / "labels.txt"
        label_f.write_text("Putamen_L\nPutamen_R\n")

        spec = AtlasSpec(source=str(f), name="test", label_file=str(label_f))
        result = load_atlas(spec)
        assert set(result.keys()) == {"Putamen_L", "Putamen_R"}


# ---------------------------------------------------------------------------
# T5 — Region statistics
# ---------------------------------------------------------------------------

class TestRegionStats:

    def _simple_subject(self):
        """5×5×5 subject with known non-zero values."""
        d = np.zeros((5, 5, 5), dtype=np.float32)
        d[0, 0, 0] = 1.0
        d[1, 1, 1] = 0.5
        d[2, 2, 2] = 0.0
        return d

    def _simple_atlas(self, subject):
        """Binary region covering first two non-zero voxels."""
        mask = np.zeros_like(subject)
        mask[0, 0, 0] = 1.0
        mask[1, 1, 1] = 1.0
        mask[2, 2, 2] = 1.0  # zero in subject
        return {"region_A": mask}

    def test_mean_overlap_known_values(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = self._simple_subject()
        atlas = self._simple_atlas(subj)
        df = compute_region_stats(subj, atlas, min_overlap_voxels=1)
        # mean of [1.0, 0.5, 0.0] = 0.5
        assert len(df) == 1
        assert pytest.approx(df.iloc[0]["mean_overlap"], abs=1e-5) == 0.5

    def test_weighted_mean_overlap(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.zeros((5, 5, 5), dtype=np.float32)
        subj[0, 0, 0] = 1.0
        subj[1, 1, 1] = 0.0
        weights = np.zeros((5, 5, 5), dtype=np.float32)
        weights[0, 0, 0] = 0.8   # high weight
        weights[1, 1, 1] = 0.2   # low weight
        atlas = {"r": weights}
        df = compute_region_stats(subj, atlas, min_overlap_voxels=1)
        # weighted_mean = (1.0*0.8 + 0.0*0.2) / (0.8+0.2) = 0.8
        assert pytest.approx(df.iloc[0]["weighted_mean_overlap"], abs=1e-5) == 0.8

    def test_weighted_mean_uniform_weights_equals_mean(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.zeros((5, 5, 5), dtype=np.float32)
        subj[0, 0, 0] = 0.6
        subj[1, 1, 1] = 0.4
        weights = np.zeros((5, 5, 5), dtype=np.float32)
        weights[0, 0, 0] = 1.0
        weights[1, 1, 1] = 1.0
        atlas = {"r": weights}
        df = compute_region_stats(subj, atlas, min_overlap_voxels=1)
        assert pytest.approx(df.iloc[0]["weighted_mean_overlap"], abs=1e-5) == \
               pytest.approx(df.iloc[0]["mean_overlap"], abs=1e-5)

    def test_fraction_covered(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.zeros((5, 5, 5), dtype=np.float32)
        subj[0, 0, 0] = 1.0
        # region has 4 voxels, only 1 non-zero in subject
        weights = np.zeros((5, 5, 5), dtype=np.float32)
        for idx in [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]:
            weights[idx] = 1.0
        df = compute_region_stats(subj, {"r": weights}, min_overlap_voxels=1)
        assert pytest.approx(df.iloc[0]["fraction_covered"], abs=1e-5) == 0.25

    def test_min_overlap_voxels_filter(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.zeros((5, 5, 5), dtype=np.float32)
        subj[0, 0, 0] = 1.0  # only 1 non-zero overlap voxel
        weights = np.zeros((5, 5, 5), dtype=np.float32)
        weights[0, 0, 0] = 1.0
        df = compute_region_stats(subj, {"r": weights}, min_overlap_voxels=2)
        assert len(df) == 0

    def test_output_sorted_descending(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.zeros((5, 5, 5), dtype=np.float32)
        subj[0, 0, 0] = 0.9
        subj[1, 1, 1] = 0.1
        w_high = np.zeros((5, 5, 5), dtype=np.float32)
        w_high[0, 0, 0] = 1.0
        w_low = np.zeros((5, 5, 5), dtype=np.float32)
        w_low[1, 1, 1] = 1.0
        df = compute_region_stats(subj, {"low": w_low, "high": w_high},
                                  min_overlap_voxels=1)
        assert df.iloc[0]["region_name"] == "high"
        assert df.iloc[1]["region_name"] == "low"

    def test_empty_atlas_returns_empty_df(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.ones((5, 5, 5), dtype=np.float32)
        df = compute_region_stats(subj, {}, min_overlap_voxels=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Subject map descriptive stats
# ---------------------------------------------------------------------------

class TestSubjectStats:

    def _make_img(self, data, affine=None):
        if affine is None:
            affine = np.eye(4)
        return nib.Nifti1Image(data.astype(np.float32), affine)

    def test_columns_present(self):
        from bcblib.tools.damage_profile._stats import compute_subject_stats
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[0, 0, 0] = 1.0
        df = compute_subject_stats(self._make_img(data))
        expected = [
            "n_nonzero_voxels", "voxel_volume_mm3", "map_volume_mm3",
            "map_sum", "map_mean_nonzero", "map_min_nonzero", "map_max",
            "map_p50_nonzero", "map_p90_nonzero", "map_p95_nonzero",
        ]
        assert list(df.columns) == expected
        assert len(df) == 1

    def test_binary_lesion_known_values(self):
        from bcblib.tools.damage_profile._stats import compute_subject_stats
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[0, 0, 0] = 1.0
        data[1, 1, 1] = 1.0
        data[2, 2, 2] = 1.0
        df = compute_subject_stats(self._make_img(data))
        assert df.iloc[0]["n_nonzero_voxels"] == 3
        assert pytest.approx(df.iloc[0]["map_volume_mm3"]) == 3.0  # 1mm³ voxels
        assert pytest.approx(df.iloc[0]["map_sum"]) == 3.0
        assert pytest.approx(df.iloc[0]["map_mean_nonzero"]) == 1.0
        assert pytest.approx(df.iloc[0]["map_min_nonzero"]) == 1.0
        assert pytest.approx(df.iloc[0]["map_max"]) == 1.0

    def test_probabilistic_map_mean_nonzero(self):
        from bcblib.tools.damage_profile._stats import compute_subject_stats
        data = np.zeros((5, 5, 5), dtype=np.float32)
        data[0, 0, 0] = 0.2
        data[1, 1, 1] = 0.8
        df = compute_subject_stats(self._make_img(data))
        assert df.iloc[0]["n_nonzero_voxels"] == 2
        assert pytest.approx(df.iloc[0]["map_mean_nonzero"]) == 0.5
        assert pytest.approx(df.iloc[0]["map_min_nonzero"]) == 0.2
        assert pytest.approx(df.iloc[0]["map_max"]) == 0.8

    def test_voxel_volume_from_affine(self):
        from bcblib.tools.damage_profile._stats import compute_subject_stats
        affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm isotropic = 8mm³ voxels
        data = np.ones((5, 5, 5), dtype=np.float32)
        df = compute_subject_stats(self._make_img(data, affine))
        assert pytest.approx(df.iloc[0]["voxel_volume_mm3"]) == 8.0
        assert pytest.approx(df.iloc[0]["map_volume_mm3"]) == 5 * 5 * 5 * 8.0

    def test_empty_map_returns_nan_stats(self):
        from bcblib.tools.damage_profile._stats import compute_subject_stats
        data = np.zeros((5, 5, 5), dtype=np.float32)
        df = compute_subject_stats(self._make_img(data))
        assert df.iloc[0]["n_nonzero_voxels"] == 0
        assert df.iloc[0]["map_volume_mm3"] == 0.0
        assert np.isnan(df.iloc[0]["map_mean_nonzero"])

    def test_subject_stats_in_damage_profile_results(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        atlas_dir = tmp_path / "atlas"
        atlas_dir.mkdir()
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(atlas_dir / "r.nii.gz", arr)
        subj = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
        spec = AtlasSpec(source=str(atlas_dir), name="a")
        results = damage_profile(subj, [spec])
        assert "_subject_map_stats" in results
        assert isinstance(results["_subject_map_stats"], pd.DataFrame)

    def test_subject_stats_csv_written(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        atlas_dir = tmp_path / "atlas"
        atlas_dir.mkdir()
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(atlas_dir / "r.nii.gz", arr)
        subj = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
        spec = AtlasSpec(source=str(atlas_dir), name="a")
        damage_profile(subj, [spec], output_dir=tmp_path / "out")
        assert (tmp_path / "out" / "subject_map_stats.csv").exists()


# ---------------------------------------------------------------------------
# T5.0 — New imaging.stats functions
# ---------------------------------------------------------------------------

class TestImagingStatsNewFunctions:

    def test_fraction_covered_array_basic(self):
        from bcblib.imaging.stats import _fraction_covered_array
        subject = np.array([1.0, 0.0, 1.0, 0.0])
        mask = np.array([1.0, 1.0, 1.0, 1.0])
        assert pytest.approx(_fraction_covered_array(subject, mask)) == 0.5

    def test_fraction_covered_array_empty_mask(self):
        from bcblib.imaging.stats import _fraction_covered_array
        subject = np.array([1.0, 1.0])
        mask = np.array([0.0, 0.0])
        assert _fraction_covered_array(subject, mask) == 0.0

    def test_fraction_covered_niftilike(self, tmp_path):
        from bcblib.imaging.stats import fraction_covered
        subj = np.zeros((4, 4, 4), dtype=np.float32)
        subj[0, 0, 0] = 1.0
        mask = np.ones((4, 4, 4), dtype=np.float32)
        sf = tmp_path / "s.nii.gz"
        mf = tmp_path / "m.nii.gz"
        _save_nifti(sf, subj)
        _save_nifti(mf, mask)
        fc = fraction_covered(str(sf), str(mf))
        assert 0.0 < fc < 1.0

    def test_weighted_region_mean_array_basic(self):
        from bcblib.imaging.stats import _weighted_region_mean_array
        subject = np.array([1.0, 0.0])
        weights = np.array([0.8, 0.2])
        # (1.0*0.8 + 0.0*0.2) / 1.0 = 0.8
        assert pytest.approx(_weighted_region_mean_array(subject, weights)) == 0.8

    def test_weighted_region_mean_zero_weights_is_nan(self):
        from bcblib.imaging.stats import _weighted_region_mean_array
        subject = np.array([1.0, 1.0])
        weights = np.array([0.0, 0.0])
        assert np.isnan(_weighted_region_mean_array(subject, weights))

    def test_weighted_region_mean_uniform_equals_mean(self):
        from bcblib.imaging.stats import _weighted_region_mean_array
        subject = np.array([0.2, 0.8])
        weights = np.array([1.0, 1.0])
        assert pytest.approx(_weighted_region_mean_array(subject, weights)) == 0.5

    def test_weighted_region_mean_niftilike(self, tmp_path):
        from bcblib.imaging.stats import weighted_region_mean
        subj = np.zeros((4, 4, 4), dtype=np.float32)
        subj[0, 0, 0] = 1.0
        w = np.zeros((4, 4, 4), dtype=np.float32)
        w[0, 0, 0] = 1.0
        sf = tmp_path / "s.nii.gz"
        wf = tmp_path / "w.nii.gz"
        _save_nifti(sf, subj)
        _save_nifti(wf, w)
        result = weighted_region_mean(str(sf), str(wf))
        assert pytest.approx(result) == 1.0


# ---------------------------------------------------------------------------
# T6 — Space handling
# ---------------------------------------------------------------------------

class TestSpaceHandling:

    def _make_img(self, shape=(5, 5, 5), affine=None):
        if affine is None:
            affine = np.eye(4)
        return _make_nifti(np.ones(shape, dtype=np.float32), affine)

    def test_same_space_no_resample(self):
        from bcblib.tools.damage_profile._space import check_and_resample
        img = self._make_img()
        with patch("bcblib.tools.damage_profile._space._check_image_space") as mock_check, \
             patch("nilearn.image.resample_to_img") as mock_nilearn:
            mock_check.return_value = {"issues": []}
            result = check_and_resample(img, img, "test_atlas")
            mock_nilearn.assert_not_called()
        assert result.shape == (5, 5, 5)

    def test_different_resolution_resamples(self):
        from bcblib.tools.damage_profile._space import check_and_resample
        subj = self._make_img(shape=(10, 10, 10))
        atlas = self._make_img(shape=(5, 5, 5))  # different shape, same world space
        resampled_fake = np.zeros((10, 10, 10), dtype=np.float32)
        fake_nii = _make_nifti(resampled_fake)

        with patch("bcblib.tools.damage_profile._space._check_image_space") as mock_check, \
             patch("nilearn.image.resample_to_img", return_value=fake_nii) as mock_nilearn:
            mock_check.return_value = {"issues": []}
            result = check_and_resample(subj, atlas, "test_atlas")
            mock_nilearn.assert_called_once()
        assert result.shape == (10, 10, 10)

    def test_affine_mismatch_error_mode(self):
        from bcblib.tools.damage_profile._space import check_and_resample
        subj = self._make_img()
        shifted = np.eye(4)
        shifted[0, 3] = 10.0
        atlas = self._make_img(affine=shifted)

        with patch("bcblib.tools.damage_profile._space._check_image_space") as mock_check:
            mock_check.return_value = {"issues": ["affine mismatch"]}
            with pytest.raises(ValueError, match="Space mismatch"):
                check_and_resample(subj, atlas, "test_atlas", on_space_mismatch="error")

    def test_affine_mismatch_warn_mode(self):
        import warnings
        from bcblib.tools.damage_profile._space import check_and_resample
        subj = self._make_img()
        shifted = np.eye(4)
        shifted[0, 3] = 10.0
        atlas = self._make_img(affine=shifted)
        fake_nii = _make_nifti(np.zeros((5, 5, 5), dtype=np.float32))

        with patch("bcblib.tools.damage_profile._space._check_image_space") as mock_check, \
             patch("nilearn.image.resample_to_img", return_value=fake_nii):
            mock_check.return_value = {"issues": ["affine mismatch"]}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = check_and_resample(
                    subj, atlas, "test_atlas", on_space_mismatch="warn"
                )
                assert any("Space mismatch" in str(warning.message) for warning in w)

    def test_cross_template_uses_templateflow(self):
        from bcblib.tools.damage_profile._space import check_and_resample
        # Subject in MNI6Asym (182×218×182), atlas in MNI2009cAsym (193×229×193)
        subj_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        subj_affine[0, 3] = 90.0
        subj = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32), subj_affine)

        atlas_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        atlas_affine[0, 3] = 96.0
        atlas = _make_nifti(np.zeros((193, 229, 193), dtype=np.float32), atlas_affine)

        fake_warped = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32), subj_affine)
        with patch(
            "bcblib.tools.damage_profile._space._apply_templateflow_warp",
            return_value=fake_warped,
        ) as mock_warp:
            result = check_and_resample(subj, atlas, "yeh")
            mock_warp.assert_called_once_with(
                atlas, "MNI152NLin2009cAsym", "MNI152NLin6Asym"
            )
        assert result.shape == (182, 218, 182)

    def test_cross_template_result_shape_matches_subject(self):
        from bcblib.tools.damage_profile._space import check_and_resample
        subj = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))
        atlas = _make_nifti(np.zeros((193, 229, 193), dtype=np.float32))
        fake_warped = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))
        with patch(
            "bcblib.tools.damage_profile._space._apply_templateflow_warp",
            return_value=fake_warped,
        ):
            result = check_and_resample(subj, atlas, "yeh")
        assert result.shape == (182, 218, 182)


# ---------------------------------------------------------------------------
# T7 — Core integration
# ---------------------------------------------------------------------------

class TestDamageProfile:

    def _make_atlas_dir(self, tmp_path, n=2, prefix="r"):
        d = tmp_path / "atlas"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = np.zeros((5, 5, 5), dtype=np.float32)
            arr[i, i, i] = 1.0
            _save_nifti(d / f"{prefix}{i}.nii.gz", arr)
        return d

    def _make_subject(self, shape=(5, 5, 5), val=1.0):
        d = np.full(shape, val, dtype=np.float32)
        return _make_nifti(d)

    def test_returns_dict_keyed_by_atlas_name(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        atlas_dir = self._make_atlas_dir(tmp_path)
        subj = self._make_subject()
        spec = AtlasSpec(source=str(atlas_dir), name="my_atlas")
        results = damage_profile(subj, [spec])
        assert "my_atlas" in results
        assert isinstance(results["my_atlas"], pd.DataFrame)

    def test_csv_written_per_atlas(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        atlas_dir = self._make_atlas_dir(tmp_path / "atlas_src")
        out_dir = tmp_path / "out"
        subj = self._make_subject()
        spec = AtlasSpec(source=str(atlas_dir), name="my_atlas")
        damage_profile(subj, [spec], output_dir=out_dir)
        assert (out_dir / "my_atlas_damage_profile.csv").exists()

    def test_accepts_path_string(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        atlas_dir = self._make_atlas_dir(tmp_path)
        subj_arr = np.ones((5, 5, 5), dtype=np.float32)
        subj_path = tmp_path / "subj.nii.gz"
        _save_nifti(subj_path, subj_arr)
        spec = AtlasSpec(source=str(atlas_dir), name="a")
        results = damage_profile(str(subj_path), [spec])
        assert "a" in results

    def test_multiple_atlases_independent(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        d1 = tmp_path / "a1"
        d1.mkdir()
        d2 = tmp_path / "a2"
        d2.mkdir()
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(d1 / "r.nii.gz", arr)
        _save_nifti(d2 / "r.nii.gz", arr)
        subj = self._make_subject()
        results = damage_profile(
            subj,
            [AtlasSpec(source=str(d1), name="a1"),
             AtlasSpec(source=str(d2), name="a2")],
        )
        assert {"a1", "a2"}.issubset(results.keys())

    def test_empty_result_when_no_overlap(self, tmp_path):
        from bcblib.tools.damage_profile import damage_profile, AtlasSpec
        atlas_dir = tmp_path / "atlas"
        atlas_dir.mkdir()
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(atlas_dir / "r.nii.gz", arr)
        # Subject is all zero
        subj = _make_nifti(np.zeros((5, 5, 5), dtype=np.float32))
        spec = AtlasSpec(source=str(atlas_dir), name="a")
        results = damage_profile(subj, [spec])
        assert len(results["a"]) == 0


# ---------------------------------------------------------------------------
# T8 — Atlas manager
# ---------------------------------------------------------------------------

class TestAtlasManager:

    def test_get_atlas_dir_returns_home_path(self):
        from bcblib.tools.damage_profile._atlas_manager import get_atlas_dir
        d = get_atlas_dir()
        assert d == Path.home() / ".bcblib" / "atlases"

    def test_list_preset_atlases(self):
        from bcblib.tools.damage_profile._atlas_manager import list_preset_atlases
        atlases = list_preset_atlases()
        assert "rojkova" in atlases
        assert "jhu_wm_prob" in atlases
        assert "yeh_hcp1065" in atlases

    def test_unknown_atlas_name_raises(self):
        from bcblib.tools.damage_profile import get_preset_atlas
        with pytest.raises(KeyError, match="Unknown preset atlas"):
            get_preset_atlas("nonexistent_atlas_xyz")

    def test_get_preset_atlas_from_cache(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import (
            get_preset_atlas, get_atlas_dir, PRESET_ATLASES,
        )
        # Pre-populate a fake cache for "rojkova" (directory format)
        cache = tmp_path / "rojkova"
        cache.mkdir()
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(cache / "tract_L.nii.gz", arr)

        with patch(
            "bcblib.tools.damage_profile._atlas_manager.get_atlas_dir",
            return_value=tmp_path,
        ):
            result = get_preset_atlas("rojkova")
        assert "tract_L" in result

    def test_get_preset_atlas_jhu_from_fsldir(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import get_preset_atlas
        jhu_dir = tmp_path / "data" / "atlases" / "JHU"
        jhu_dir.mkdir(parents=True)
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        # JHU wm_prob is a 4D NIfTI — use a minimal one with one volume
        data_4d = arr[..., np.newaxis]
        _save_nifti(jhu_dir / "JHU-ICBM-tracts-prob-2mm.nii.gz", data_4d)

        with patch.dict(os.environ, {"FSLDIR": str(tmp_path)}), \
             patch(
                 "bcblib.tools.damage_profile._atlas_manager.get_atlas_dir",
                 return_value=tmp_path / "empty_cache",
             ):
            result = get_preset_atlas("jhu_wm_prob")
        assert len(result) >= 1

    def test_download_prompts_user(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import get_preset_atlas

        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0

        def fake_download(info, dest):
            dest.mkdir(parents=True, exist_ok=True)
            _save_nifti(dest / "tract.nii.gz", arr)

        with patch(
            "bcblib.tools.damage_profile._atlas_manager.get_atlas_dir",
            return_value=tmp_path,
        ), patch(
            "bcblib.tools.damage_profile._atlas_manager._download_atlas",
            side_effect=fake_download,
        ), patch("builtins.input", return_value="y"):
            result = get_preset_atlas("rojkova")
        assert "tract" in result

    def test_download_skipped_when_user_says_no(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import get_preset_atlas
        with patch(
            "bcblib.tools.damage_profile._atlas_manager.get_atlas_dir",
            return_value=tmp_path,
        ), patch("builtins.input", return_value="n"):
            with pytest.raises(RuntimeError, match="declined"):
                get_preset_atlas("rojkova")

    def test_assume_yes_skips_prompt(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import get_preset_atlas

        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0

        def fake_download(info, dest):
            dest.mkdir(parents=True, exist_ok=True)
            _save_nifti(dest / "tract.nii.gz", arr)

        with patch(
            "bcblib.tools.damage_profile._atlas_manager.get_atlas_dir",
            return_value=tmp_path,
        ), patch(
            "bcblib.tools.damage_profile._atlas_manager._download_atlas",
            side_effect=fake_download,
        ), patch("builtins.input") as mock_input:
            result = get_preset_atlas("rojkova", assume_yes=True)
            mock_input.assert_not_called()
        assert "tract" in result


# ---------------------------------------------------------------------------
# T9 — CLI
# ---------------------------------------------------------------------------

class TestCLI:

    def _make_subject_file(self, tmp_path):
        arr = np.ones((5, 5, 5), dtype=np.float32)
        p = tmp_path / "subj.nii.gz"
        _save_nifti(p, arr)
        return p

    def _make_atlas_dir(self, tmp_path):
        d = tmp_path / "atlas"
        d.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(d / "r0.nii.gz", arr)
        return d

    def test_parse_args_minimal(self, tmp_path):
        from bcblib.scripts.run_damage_profile import parse_args
        subj = self._make_subject_file(tmp_path)
        atlas = self._make_atlas_dir(tmp_path)
        args = parse_args([
            "--map", str(subj),
            "--atlas", str(atlas),
            "--name", "test_atlas",
        ])
        assert args.map == str(subj)
        assert args.atlas_paths == [str(atlas)]
        assert args.atlas_names == ["test_atlas"]

    def test_parse_args_preset(self, tmp_path):
        from bcblib.scripts.run_damage_profile import parse_args
        subj = self._make_subject_file(tmp_path)
        args = parse_args([
            "--map", str(subj),
            "--preset", "rojkova",
            "--assume-yes",
        ])
        assert args.presets == ["rojkova"]
        assert args.assume_yes is True

    def test_parse_args_missing_atlas_raises(self, tmp_path):
        from bcblib.scripts.run_damage_profile import parse_args
        subj = self._make_subject_file(tmp_path)
        with pytest.raises(SystemExit):
            parse_args(["--map", str(subj)])

    def test_parse_args_atlas_name_count_mismatch_raises(self, tmp_path):
        from bcblib.scripts.run_damage_profile import parse_args
        subj = self._make_subject_file(tmp_path)
        atlas = self._make_atlas_dir(tmp_path)
        with pytest.raises(SystemExit):
            parse_args([
                "--map", str(subj),
                "--atlas", str(atlas),
                # missing --name
            ])

    def test_main_writes_csv(self, tmp_path):
        from bcblib.scripts.run_damage_profile import main
        subj = self._make_subject_file(tmp_path)
        atlas = self._make_atlas_dir(tmp_path)
        out_dir = tmp_path / "out"
        ret = main([
            "--map", str(subj),
            "--atlas", str(atlas),
            "--name", "my_atlas",
            "--output-dir", str(out_dir),
        ])
        assert ret == 0
        assert (out_dir / "my_atlas_damage_profile.csv").exists()


# ---------------------------------------------------------------------------
# Coverage gap closers
# ---------------------------------------------------------------------------

class TestCoverageGaps:
    """Targeted tests to cover error paths and rarely-exercised branches."""

    # --- _atlas.py ---

    def test_detect_atlas_format_invalid_raises(self, tmp_path):
        from bcblib.tools.damage_profile._atlas import detect_atlas_format
        bad = tmp_path / "not_a_nifti.txt"
        bad.write_text("garbage")
        with pytest.raises(ValueError, match="Cannot determine atlas format"):
            detect_atlas_format(str(bad))

    def test_parse_fsl_xml_labels(self):
        from bcblib.tools.damage_profile._atlas import _parse_fsl_xml_labels
        xml = (
            '<atlas><data>'
            '<label index="0">CST_L</label>'
            '<label index="1">CST_R</label>'
            '</data></atlas>'
        )
        result = _parse_fsl_xml_labels(xml)
        assert result == {0: "CST_L", 1: "CST_R"}

    def test_parse_fsl_xml_labels_invalid_index_skipped(self):
        from bcblib.tools.damage_profile._atlas import _parse_fsl_xml_labels
        xml = '<atlas><data><label index="x">Bad</label></data></atlas>'
        assert _parse_fsl_xml_labels(xml) == {}

    def test_parse_label_file_xml_dispatch(self, tmp_path):
        from bcblib.tools.damage_profile._atlas import _parse_label_file
        f = tmp_path / "labels.xml"
        f.write_text(
            '<atlas><data>'
            '<label index="1">Region_A</label>'
            '</data></atlas>'
        )
        result = _parse_label_file(str(f))
        assert result == {1: "Region_A"}

    def test_parse_text_labels_tsv(self, tmp_path):
        from bcblib.tools.damage_profile._atlas import _parse_text_labels
        text = "1\tPutamen_L\n2\tPutamen_R\n"
        result = _parse_text_labels(text)
        assert result == {1: "Putamen_L", 2: "Putamen_R"}

    def test_parse_text_labels_blank_lines_skipped(self):
        from bcblib.tools.damage_profile._atlas import _parse_text_labels
        text = "RegionA\n\nRegionB\n"
        result = _parse_text_labels(text)
        assert 1 in result and result[1] == "RegionA"
        # blank line skipped; RegionB gets index 3 (line 3 in file, but blank
        # shifts enumeration) — what matters is blank line doesn't error
        assert len(result) == 2

    def test_4d_nifti_threshold_excludes_volume(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5, 2), dtype=np.float32)
        data[0, 0, 0, 0] = 0.3   # below threshold 0.5 → all zeroed → excluded
        data[1, 1, 1, 1] = 1.0   # above threshold → kept
        f = tmp_path / "a.nii.gz"
        _save_nifti(f, data)
        spec = AtlasSpec(source=str(f), name="t", threshold=0.5)
        result = load_atlas(spec)
        assert len(result) == 1

    def test_4d_nifti_empty_volume_excluded(self, tmp_path):
        from bcblib.tools.damage_profile import AtlasSpec, load_atlas
        data = np.zeros((5, 5, 5, 2), dtype=np.float32)
        data[0, 0, 0, 0] = 1.0   # non-empty
        # volume 1 stays all-zero → excluded
        f = tmp_path / "a.nii.gz"
        _save_nifti(f, data)
        spec = AtlasSpec(source=str(f), name="t")
        result = load_atlas(spec)
        assert len(result) == 1

    # --- _atlas_manager.py ---

    def test_get_preset_atlas_explicit_path(self, tmp_path):
        from bcblib.tools.damage_profile import get_preset_atlas
        d = tmp_path / "atlas"
        d.mkdir()
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        _save_nifti(d / "r.nii.gz", arr)
        result = get_preset_atlas("rojkova", path=str(d))
        assert "r" in result

    def test_get_preset_atlas_jhu_no_fsldir_raises(self, tmp_path):
        from bcblib.tools.damage_profile import get_preset_atlas
        import os
        env = {k: v for k, v in os.environ.items() if k != "FSLDIR"}
        with patch(
            "bcblib.tools.damage_profile._atlas_manager.get_atlas_dir",
            return_value=tmp_path / "empty",
        ), patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="JHU atlas not found"):
                get_preset_atlas("jhu_wm_prob")

    def test_download_atlas_zip(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import _download_atlas, PRESET_ATLASES
        import io, zipfile as zf_mod
        info = PRESET_ATLASES["rojkova"]

        # Build a tiny in-memory zip to return from urlretrieve
        buf = io.BytesIO()
        with zf_mod.ZipFile(buf, "w") as z:
            z.writestr("dummy.txt", "data")
        zip_bytes = buf.getvalue()

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(zip_bytes)

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            _download_atlas(info, tmp_path)

        assert (tmp_path / "dummy.txt").exists()

    def test_download_atlas_no_url_raises(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import _download_atlas, AtlasInfo
        info = AtlasInfo(
            full_name="NoURL Atlas", url=None, size_mb=0,
            fmt="directory", space="MNI152NLin6Asym", citation=""
        )
        with pytest.raises(ValueError, match="no download URL"):
            _download_atlas(info, tmp_path)

    def test_download_atlas_single_file(self, tmp_path):
        from bcblib.tools.damage_profile._atlas_manager import _download_atlas, AtlasInfo
        info = AtlasInfo(
            full_name="Single File Atlas",
            url="https://example.com/atlas.nii",
            size_mb=1.0, fmt="label_nifti",
            space="MNI152NLin6Asym", citation=""
        )

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"fake")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            _download_atlas(info, tmp_path)
        assert (tmp_path / "atlas.nii").exists()

    # --- _core.py ---

    def test_load_atlas_reference_from_nifti_file(self, tmp_path):
        from bcblib.tools.damage_profile._core import _load_atlas_reference
        from bcblib.tools.damage_profile import AtlasSpec
        arr = np.zeros((5, 5, 5), dtype=np.float32)
        f = tmp_path / "atlas.nii.gz"
        _save_nifti(f, arr)
        spec = AtlasSpec(source=str(f), name="a")
        ref = _load_atlas_reference(spec)
        assert ref is not None
        assert ref.shape == (5, 5, 5)

    def test_resample_atlas_dict_when_needed(self, tmp_path):
        from bcblib.tools.damage_profile._core import _resample_atlas_dict
        import nibabel as nib
        # atlas 5×5×5 with identity affine; subject 5×5×5 with shifted affine
        atlas_affine = np.eye(4)
        subj_affine = np.eye(4)
        subj_affine[0, 3] = 20.0  # shift forces resample
        subject_img = _make_nifti(np.zeros((5, 5, 5), dtype=np.float32), subj_affine)
        atlas_dict = {"r": np.ones((5, 5, 5), dtype=np.float32)}

        fake_nii = _make_nifti(np.zeros((5, 5, 5), dtype=np.float32), subj_affine)
        with patch("bcblib.tools.damage_profile._space.check_and_resample",
                   return_value=np.zeros((5, 5, 5), dtype=np.float32)) as mock_cr, \
             patch("bcblib.tools.damage_profile._core.check_and_resample",
                   return_value=np.zeros((5, 5, 5), dtype=np.float32)):
            result = _resample_atlas_dict(
                atlas_dict, atlas_affine, subject_img, None, "warn"
            )
        assert "r" in result

    # --- _stats.py ---

    def test_compute_region_stats_empty_mask_skipped(self):
        from bcblib.tools.damage_profile._stats import compute_region_stats
        subj = np.ones((5, 5, 5), dtype=np.float32)
        weights = np.zeros((5, 5, 5), dtype=np.float32)  # all-zero mask
        df = compute_region_stats(subj, {"empty_region": weights})
        assert len(df) == 0

    # --- _space.py ---

    def test_apply_templateflow_warp_no_warp_raises(self):
        from bcblib.tools.damage_profile._space import _apply_templateflow_warp
        atlas = _make_nifti(np.zeros((5, 5, 5), dtype=np.float32))
        with patch("bcblib.tools.damage_profile._space.tflow.get", return_value=None):
            with pytest.raises(RuntimeError, match="TemplateFlow has no warp"):
                _apply_templateflow_warp(atlas, "MNI152NLin2009cAsym", "MNI152NLin6Asym")
