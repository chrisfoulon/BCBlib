"""Tests for the input-validation helpers added to bcblib.tools.best_overlap.

These tests cover:
- _write_diagnostic
- _check_image_space (shape / affine / orientation, error vs warn modes)
- compute_cluster_features (shape guard, empty-overlap path)
- run_connectivity_analysis (empty-overlap diagnostic file, space-mismatch
  propagation) — Bayesian sampling is bypassed via monkeypatching so tests
  run in milliseconds.

No external data are required; all NIfTI images are created synthetically
in tmp_path.
"""

import json
import os
import textwrap
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bcblib.tools.best_overlap import (
    _check_image_space,
    _write_diagnostic,
    compute_cluster_features,
    run_connectivity_analysis,
)


# ---------------------------------------------------------------------------
# Tiny NIfTI factories
# ---------------------------------------------------------------------------

def _make_img(shape=(10, 10, 10), voxel_size=2.0, data=None, dtype=np.float32):
    """Return an in-memory Nifti1Image."""
    if data is None:
        data = np.zeros(shape, dtype=dtype)
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    return nib.Nifti1Image(data.astype(dtype), affine)


def _save_img(img, path):
    nib.save(img, str(path))
    return path


def _make_parcellation(tmp_path, shape=(10, 10, 10), voxel_size=2.0, n_regions=3):
    """Save a label image with *n_regions* non-zero integer regions."""
    data = np.zeros(shape, dtype=np.int32)
    # Carve out n_regions equal-sized blocks along the first axis
    step = shape[0] // n_regions
    for i in range(n_regions):
        data[i * step:(i + 1) * step, :, :] = i + 1
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data, dtype=np.int32)
    p = tmp_path / "parc.nii.gz"
    _save_img(img, p)
    return p


def _make_preserved_nonzero(tmp_path, shape=(10, 10, 10), voxel_size=2.0, name="preserved.nii.gz"):
    """Save a preserved-connection map that overlaps with every region."""
    data = np.ones(shape, dtype=np.float32) * 0.5
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data)
    p = tmp_path / name
    _save_img(img, p)
    return p


# ---------------------------------------------------------------------------
# _write_diagnostic
# ---------------------------------------------------------------------------

class TestWriteDiagnostic:
    def test_creates_file_with_content(self, tmp_path):
        path = str(tmp_path / "diag.txt")
        lines = ["Line 1", "Line 2", "Line 3"]
        _write_diagnostic(path, lines)
        text = Path(path).read_text()
        for line in lines:
            assert line in text

    def test_none_path_does_not_create_file(self, tmp_path, capsys):
        lines = ["only to console"]
        _write_diagnostic(None, lines)
        out = capsys.readouterr().out
        assert "only to console" in out
        # No file should have been written in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_console_output_matches_lines(self, capsys):
        lines = ["alpha", "beta"]
        _write_diagnostic(None, lines)
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out


# ---------------------------------------------------------------------------
# _check_image_space
# ---------------------------------------------------------------------------

class TestCheckImageSpace:
    def test_matching_images_no_error(self):
        img = _make_img(shape=(8, 8, 8), voxel_size=2.0)
        info = _check_image_space(img, "a", img, "b")
        assert info["issues"] == []

    def test_shape_mismatch_always_raises(self):
        img_a = _make_img(shape=(8, 8, 8))
        img_b = _make_img(shape=(9, 8, 8))
        with pytest.raises(ValueError, match="SHAPE"):
            _check_image_space(img_a, "a", img_b, "b", on_mismatch="warn")

    def test_affine_mismatch_error_mode_raises(self):
        # Use voxel_size=5.0 so the diagonal diff (3.0) clearly exceeds atol=1.0
        img_a = _make_img(shape=(8, 8, 8), voxel_size=2.0)
        img_b = _make_img(shape=(8, 8, 8), voxel_size=5.0)
        with pytest.raises(ValueError, match="[Aa]ffine"):
            _check_image_space(img_a, "a", img_b, "b", on_mismatch="error")

    def test_affine_mismatch_warn_mode_no_raise(self, capsys):
        # Use voxel_size=5.0 so the diagonal diff (3.0) clearly exceeds atol=1.0
        img_a = _make_img(shape=(8, 8, 8), voxel_size=2.0)
        img_b = _make_img(shape=(8, 8, 8), voxel_size=5.0)
        info = _check_image_space(img_a, "a", img_b, "b", on_mismatch="warn")
        assert len(info["issues"]) >= 1
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_atol_is_subvoxel_for_small_voxels(self):
        """For 0.5 mm voxels, atol should be 0.5 (sub-voxel wins over 1 mm cap)."""
        img = _make_img(shape=(8, 8, 8), voxel_size=0.5)
        info = _check_image_space(img, "a", img, "b")
        assert info["atol"] == pytest.approx(0.5)

    def test_atol_is_1mm_for_large_voxels(self):
        """For 3 mm voxels, atol should be capped at 1.0 mm (submm cap wins)."""
        img = _make_img(shape=(8, 8, 8), voxel_size=3.0)
        info = _check_image_space(img, "a", img, "b")
        assert info["atol"] == pytest.approx(1.0)

    def test_orientation_mismatch_error_mode_raises(self):
        """Flipped affine (LAS vs RAS) should trigger orientation mismatch."""
        shape = (8, 8, 8)
        data = np.zeros(shape, dtype=np.float32)
        affine_ras = np.diag([2.0, 2.0, 2.0, 1.0])
        affine_las = np.diag([-2.0, 2.0, 2.0, 1.0])  # flipped X → LAS
        img_ras = nib.Nifti1Image(data, affine_ras)
        img_las = nib.Nifti1Image(data, affine_las)
        with pytest.raises(ValueError, match="[Oo]rientation|[Aa]ffine"):
            _check_image_space(img_ras, "ras", img_las, "las", on_mismatch="error")

    def test_orientation_mismatch_warn_mode_reports_issue(self, capsys):
        shape = (8, 8, 8)
        data = np.zeros(shape, dtype=np.float32)
        affine_ras = np.diag([2.0, 2.0, 2.0, 1.0])
        affine_las = np.diag([-2.0, 2.0, 2.0, 1.0])
        img_ras = nib.Nifti1Image(data, affine_ras)
        img_las = nib.Nifti1Image(data, affine_las)
        info = _check_image_space(img_ras, "ras", img_las, "las", on_mismatch="warn")
        assert len(info["issues"]) >= 1
        capsys.readouterr()  # consume output

    def test_info_dict_has_expected_keys(self):
        img = _make_img()
        info = _check_image_space(img, "a", img, "b")
        for key in ("shape_a", "shape_b", "affine_a", "affine_b",
                    "atol", "orientation_a", "orientation_b", "issues"):
            assert key in info

    def test_4d_image_uses_spatial_shape(self):
        """Shape check should only compare the first 3 dims."""
        data_3d = np.zeros((8, 8, 8), dtype=np.float32)
        data_4d = np.zeros((8, 8, 8, 5), dtype=np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img_3d = nib.Nifti1Image(data_3d, affine)
        img_4d = nib.Nifti1Image(data_4d, affine)
        # Should NOT raise: spatial dims match
        info = _check_image_space(img_3d, "3d", img_4d, "4d")
        assert info["issues"] == []


# ---------------------------------------------------------------------------
# compute_cluster_features — shape guard
# ---------------------------------------------------------------------------

class TestComputeClusterFeaturesGuard:
    def test_shape_mismatch_raises_descriptive_error(self, tmp_path):
        parc_path = _make_parcellation(tmp_path, shape=(10, 10, 10))
        # Pass a preserved map with a different shape
        preserved = np.ones((8, 8, 8), dtype=np.float32) * 0.5
        with pytest.raises(ValueError, match="[Ss]hape mismatch"):
            compute_cluster_features(preserved, str(parc_path))

    def test_empty_overlap_returns_empty_dataframe(self, tmp_path):
        parc_path = _make_parcellation(tmp_path, shape=(10, 10, 10))
        # All-zero preserved map → no overlap
        preserved = np.zeros((10, 10, 10), dtype=np.float32)
        df = compute_cluster_features(preserved, str(parc_path))
        assert df.empty

    def test_nonzero_overlap_returns_rows(self, tmp_path):
        parc_path = _make_parcellation(tmp_path, shape=(10, 10, 10), n_regions=3)
        preserved = np.ones((10, 10, 10), dtype=np.float32) * 0.5
        df = compute_cluster_features(preserved, str(parc_path))
        assert len(df) == 3
        assert "SumProb" in df.columns


# ---------------------------------------------------------------------------
# run_connectivity_analysis — validation integration tests
# (Bayesian sampling hard-bypassed via monkeypatch)
# ---------------------------------------------------------------------------

def _make_minimal_cluster_img(tmp_path, shape, voxel_size, val=0.8, name="cluster.nii.gz"):
    data = np.full(shape, val, dtype=np.float32)
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data)
    p = tmp_path / name
    _save_img(img, p)
    return p


def _make_minimal_patient_img(tmp_path, shape, voxel_size, val=0.1, name="patient.nii.gz"):
    data = np.full(shape, val, dtype=np.float32)
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data)
    p = tmp_path / name
    _save_img(img, p)
    return p


def _mock_run_bayesian_model(df):
    """Minimal stub that returns the structure expected by generate_report."""
    import pandas as pd
    import numpy as np

    n = len(df)
    # Build a fake InferenceData-like object using a plain namespace
    class FakeTrace:
        class posterior:
            mu_det = None  # replaced below
        class posterior_predictive:
            obs = None  # replaced below
        class log_likelihood:
            obs = None

    # arviz stores things as xarray-like; generate_report accesses:
    #   trace.posterior["mu_det"].stack(sample=...).values
    # We use a real numpy array wrapped in a tiny class.
    import xarray as xr

    coords = {"chain": [0], "draw": list(range(10)), "mu_det_dim_0": list(range(n))}
    mu_data = np.ones((1, 10, n)) * 0.5  # shape: chain, draw, obs
    mu_da = xr.DataArray(mu_data, dims=["chain", "draw", "mu_det_dim_0"])

    obs_data = np.ones((1, 10, n)) * 0.5
    obs_da = xr.DataArray(obs_data, dims=["chain", "draw", "obs_dim_0"])

    trace = type("FakeTrace", (), {
        "posterior": type("P", (), {
            "__getitem__": staticmethod(lambda k: mu_da if k == "mu_det" else None),
            "beta_cv": xr.DataArray(np.array([0.1])),
            "beta_p90": xr.DataArray(np.array([0.1])),
            "beta_ro": xr.DataArray(np.array([0.1])),
            "beta_de": xr.DataArray(np.array([0.1])),
            "beta_wc": xr.DataArray(np.array([0.1])),
        })(),
        "posterior_predictive": type("PP", (), {
            "__getitem__": staticmethod(lambda k: obs_da),
        })(),
    })()

    return {
        "trace": trace,
        "bayesian_r2": 0.5,
        "loo": None,
        "effect_sizes": {k: 0.1 for k in [
            "log_ClusterVolume", "P90", "RatioOverlap",
            "DensityOverlap", "WeightedClusterContribution",
        ]},
        "standardization_params": {},
    }


class TestRunConnectivityAnalysisValidation:

    def test_empty_overlap_skip_writes_diagnostic(self, tmp_path, monkeypatch):
        """
        When the preserved map has no overlap with the parcellation (e.g. the
        patient disconnects everything), the script should skip the patient,
        write a diagnostic file, and NOT crash.
        """
        shape = (10, 10, 10)
        # Patient disconnects everything (patient_prob ≈ 1 → preserved ≈ 0)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=1.0)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )

        # Should not raise
        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        diag_files = list(out.glob("*_FAILED_diagnostic.txt"))
        assert len(diag_files) == 1, "Expected one diagnostic file"
        text = diag_files[0].read_text()
        assert "ANALYSIS FAILED" in text
        assert "POSSIBLE CAUSES" in text
        # No report CSV should have been written
        assert not list(out.glob("*_analysis_report.csv"))

    def test_empty_overlap_diagnostic_contains_input_info(self, tmp_path, monkeypatch):
        shape = (10, 10, 10)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=1.0, name="p1.nii.gz")
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        run_connectivity_analysis([str(patient)], [str(cluster)], str(parc), str(out))

        diag_files = list(out.glob("*_FAILED_diagnostic.txt"))
        text = diag_files[0].read_text()
        # Should mention shapes, n regions, n nonzero
        assert "(10, 10, 10)" in text
        assert "3" in text  # n_regions

    def test_multiple_patients_empty_and_valid(self, tmp_path, monkeypatch):
        """
        With two patients — one with full disconnection (empty overlap) and one
        with non-zero overlap — only the failing patient should produce a
        diagnostic file; the passing patient should produce a report.
        """
        import xarray as xr

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)

        # Patient 1: everything disconnected → zero preserved
        p_fail = tmp_path / "pat_fail.nii.gz"
        _save_img(_make_img(shape=shape, voxel_size=2.0,
                            data=np.ones(shape, dtype=np.float32)), p_fail)

        # Patient 2: mild disconnection → non-zero preserved
        p_ok = tmp_path / "pat_ok.nii.gz"
        _save_img(_make_img(shape=shape, voxel_size=2.0,
                            data=np.full(shape, 0.1, dtype=np.float32)), p_ok)

        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        # Also stub generate_report so we don't need a real trace
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv: Path(output_csv).write_text("ok\n"),
        )

        run_connectivity_analysis(
            [str(p_fail), str(p_ok)], [str(cluster)], str(parc), str(out)
        )

        diag_files = list(out.glob("*_FAILED_diagnostic.txt"))
        assert len(diag_files) == 1
        assert "pat_fail" in diag_files[0].name

        report_files = list(out.glob("*_analysis_report.csv"))
        assert len(report_files) == 1
        assert "pat_ok" in report_files[0].name

    def test_shape_mismatch_patient_cluster_raises(self, tmp_path, monkeypatch):
        """Shape mismatch between patient and cluster always raises ValueError."""
        parc = _make_parcellation(tmp_path, shape=(10, 10, 10), voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, (10, 10, 10), voxel_size=2.0)
        cluster = _make_minimal_cluster_img(tmp_path, (12, 12, 12), voxel_size=2.0)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        with pytest.raises(ValueError, match="[Ss]hape"):
            run_connectivity_analysis(
                [str(patient)], [str(cluster)], str(parc), str(out)
            )

    def test_shape_mismatch_patient_parcellation_raises(self, tmp_path, monkeypatch):
        """Shape mismatch between patient and parcellation always raises ValueError."""
        parc = _make_parcellation(tmp_path, shape=(12, 12, 12), voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, (10, 10, 10), voxel_size=2.0)
        cluster = _make_minimal_cluster_img(tmp_path, (10, 10, 10), voxel_size=2.0)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        with pytest.raises(ValueError, match="[Ss]hape"):
            run_connectivity_analysis(
                [str(patient)], [str(cluster)], str(parc), str(out)
            )

    def test_affine_mismatch_error_mode_raises(self, tmp_path, monkeypatch):
        shape = (10, 10, 10)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0)
        # Cluster differs by 3mm per voxel → clearly > atol=1.0
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=5.0)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        with pytest.raises(ValueError, match="[Aa]ffine|[Oo]rientation"):
            run_connectivity_analysis(
                [str(patient)], [str(cluster)], str(parc), str(out),
                on_space_mismatch="error",
            )

    def test_affine_mismatch_warn_mode_continues(self, tmp_path, monkeypatch, capsys):
        """
        With on_space_mismatch='warn', an affine mismatch should print a warning
        but not abort.  Because the shapes match, computation proceeds (even if
        the result is spatially wrong — that is now the user's responsibility).
        """
        shape = (10, 10, 10)
        # patient voxel_size=2, cluster voxel_size=5 → diff 3mm > atol=1.0
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=5.0, val=0.8)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv: Path(output_csv).write_text("ok\n"),
        )
        # Should NOT raise
        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out),
            on_space_mismatch="warn",
        )
        out_text = capsys.readouterr().out
        assert "WARNING" in out_text

    def test_per_cluster_empty_warning_printed(self, tmp_path, monkeypatch, capsys):
        """When a single cluster yields zero overlap, a per-cluster WARNING is printed."""
        shape = (10, 10, 10)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=1.0)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        run_connectivity_analysis([str(patient)], [str(cluster)], str(parc), str(out))
        out_text = capsys.readouterr().out
        assert "WARNING" in out_text  # per-cluster warning

    def test_numpy_shape_error_wrapped_with_context(self, tmp_path, monkeypatch):
        """
        If numpy raises a broadcast error during preserved = cluster * (1 - patient),
        it should be caught and re-raised with context.
        This is hard to trigger via the space-check path (which fires first), so we
        bypass _check_image_space and test the try/except directly by triggering the
        shape mismatch after the check.
        """
        shape = (10, 10, 10)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0)
        out = tmp_path / "out"

        # Bypass _check_image_space so the try/except around the multiply is exercised
        monkeypatch.setattr(
            "bcblib.tools.best_overlap._check_image_space",
            lambda *args, **kwargs: {"issues": []},
        )

        # Make get_fdata() of the cluster return the wrong shape
        original_load = nib.load
        call_count = [0]

        def patched_load(path):
            img = original_load(path)
            call_count[0] += 1
            # 3rd load is the cluster inside the loop (1=patient, 2=parc, 3=cluster)
            if call_count[0] == 3:
                wrong_data = np.ones((9, 9, 9), dtype=np.float32)
                return nib.Nifti1Image(wrong_data, img.affine)
            return img

        monkeypatch.setattr(nib, "load", patched_load)
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )

        with pytest.raises(ValueError, match="[Cc]ould not compute preserved|[Ss]hape"):
            run_connectivity_analysis(
                [str(patient)], [str(cluster)], str(parc), str(out)
            )
