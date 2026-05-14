"""Tests for the input-validation helpers added to bcblib.tools.best_overlap.

These tests cover:
- _write_diagnostic
- _check_image_space (shape / affine / orientation, error vs warn modes)
- compute_cluster_features (shape guard, empty-overlap path)
- compute_rope_classification (all three categories, edge cases)
- generate_model_summary (T1: actual newlines, not literal \\n)
- generate_report (T6: dual CSV output, ROPE column present)
- run_connectivity_analysis (empty-overlap diagnostic file, space-mismatch
  propagation, detection-limit filters) — Bayesian sampling is bypassed via
  monkeypatching so tests run in milliseconds.

No external data are required; all NIfTI images are created synthetically
in tmp_path.
"""

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
    compute_rope_classification,
    generate_model_summary,
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
    step = shape[0] // n_regions
    for i in range(n_regions):
        data[i * step:(i + 1) * step, :, :] = i + 1
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data, dtype=np.int32)
    p = tmp_path / "parc.nii.gz"
    _save_img(img, p)
    return p


def _make_preserved_nonzero(tmp_path, shape=(10, 10, 10), voxel_size=2.0,
                             name="preserved.nii.gz"):
    """Save a preserved-connection map that overlaps with every region."""
    data = np.ones(shape, dtype=np.float32) * 0.5
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data)
    p = tmp_path / name
    _save_img(img, p)
    return p


# ---------------------------------------------------------------------------
# Minimal pyplot stub — avoids opening a display in headless CI environments
# ---------------------------------------------------------------------------

class _MockPlt:
    """Drop-in stub for matplotlib.pyplot that discards all drawing calls."""
    @staticmethod
    def figure(*a, **kw): return None
    @staticmethod
    def errorbar(*a, **kw): return None
    @staticmethod
    def xticks(*a, **kw): return None
    @staticmethod
    def xlabel(*a, **kw): return None
    @staticmethod
    def ylabel(*a, **kw): return None
    @staticmethod
    def title(*a, **kw): return None
    @staticmethod
    def tight_layout(*a, **kw): return None
    @staticmethod
    def savefig(*a, **kw): return None
    @staticmethod
    def close(*a, **kw): return None


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
        img_a = _make_img(shape=(8, 8, 8), voxel_size=2.0)
        img_b = _make_img(shape=(8, 8, 8), voxel_size=5.0)
        with pytest.raises(ValueError, match="[Aa]ffine"):
            _check_image_space(img_a, "a", img_b, "b", on_mismatch="error")

    def test_affine_mismatch_warn_mode_no_raise(self, capsys):
        img_a = _make_img(shape=(8, 8, 8), voxel_size=2.0)
        img_b = _make_img(shape=(8, 8, 8), voxel_size=5.0)
        info = _check_image_space(img_a, "a", img_b, "b", on_mismatch="warn")
        assert len(info["issues"]) >= 1
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_atol_is_subvoxel_for_small_voxels(self):
        img = _make_img(shape=(8, 8, 8), voxel_size=0.5)
        info = _check_image_space(img, "a", img, "b")
        assert info["atol"] == pytest.approx(0.5)

    def test_atol_is_1mm_for_large_voxels(self):
        img = _make_img(shape=(8, 8, 8), voxel_size=3.0)
        info = _check_image_space(img, "a", img, "b")
        assert info["atol"] == pytest.approx(1.0)

    def test_orientation_mismatch_error_mode_raises(self):
        shape = (8, 8, 8)
        data = np.zeros(shape, dtype=np.float32)
        affine_ras = np.diag([2.0, 2.0, 2.0, 1.0])
        affine_las = np.diag([-2.0, 2.0, 2.0, 1.0])
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
        capsys.readouterr()

    def test_info_dict_has_expected_keys(self):
        img = _make_img()
        info = _check_image_space(img, "a", img, "b")
        for key in ("shape_a", "shape_b", "affine_a", "affine_b",
                    "atol", "orientation_a", "orientation_b", "issues"):
            assert key in info

    def test_4d_image_uses_spatial_shape(self):
        data_3d = np.zeros((8, 8, 8), dtype=np.float32)
        data_4d = np.zeros((8, 8, 8, 5), dtype=np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img_3d = nib.Nifti1Image(data_3d, affine)
        img_4d = nib.Nifti1Image(data_4d, affine)
        info = _check_image_space(img_3d, "3d", img_4d, "4d")
        assert info["issues"] == []


# ---------------------------------------------------------------------------
# compute_cluster_features — shape guard
# ---------------------------------------------------------------------------

class TestComputeClusterFeaturesGuard:
    def test_shape_mismatch_raises_descriptive_error(self, tmp_path):
        parc_path = _make_parcellation(tmp_path, shape=(10, 10, 10))
        preserved = np.ones((8, 8, 8), dtype=np.float32) * 0.5
        with pytest.raises(ValueError, match="[Ss]hape mismatch"):
            compute_cluster_features(preserved, str(parc_path))

    def test_empty_overlap_returns_empty_dataframe(self, tmp_path):
        parc_path = _make_parcellation(tmp_path, shape=(10, 10, 10))
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
# compute_rope_classification — unit tests (T5.7)
# ---------------------------------------------------------------------------

class TestComputeRopeClassification:

    def test_meaningful_ci_clearly_above_rope(self):
        assert compute_rope_classification(1.0, 2.0, rope_high=0.5) == "Meaningful"

    def test_negligible_ci_entirely_within_rope(self):
        assert compute_rope_classification(-1.0, 0.3, rope_high=0.5) == "Negligible"

    def test_inconclusive_ci_straddles_boundary(self):
        assert compute_rope_classification(-0.5, 1.0, rope_high=0.5) == "Inconclusive"

    def test_edge_ci_lower_exactly_at_rope_high_is_inconclusive(self):
        # ci_lower == rope_high: condition ci_lower > rope_high is False
        assert compute_rope_classification(0.5, 1.0, rope_high=0.5) == "Inconclusive"

    def test_edge_ci_upper_exactly_at_rope_high_is_negligible(self):
        # ci_upper == rope_high: condition ci_upper <= rope_high is True
        assert compute_rope_classification(-0.5, 0.5, rope_high=0.5) == "Negligible"

    def test_negative_rope_high_meaningful(self):
        assert compute_rope_classification(0.0, 1.0, rope_high=-0.5) == "Meaningful"

    def test_negative_rope_high_negligible(self):
        assert compute_rope_classification(-2.0, -0.6, rope_high=-0.5) == "Negligible"

    def test_negative_rope_high_inconclusive(self):
        assert compute_rope_classification(-0.7, 0.5, rope_high=-0.5) == "Inconclusive"

    def test_zero_width_ci_above_rope(self):
        assert compute_rope_classification(1.0, 1.0, rope_high=0.5) == "Meaningful"

    def test_zero_width_ci_at_rope_boundary(self):
        # Both bounds equal rope_high: ci_lower == rope_high → not Meaningful;
        # ci_upper == rope_high → Negligible
        assert compute_rope_classification(0.5, 0.5, rope_high=0.5) == "Negligible"


# ---------------------------------------------------------------------------
# generate_model_summary — T1: actual newlines (not literal \\n)
# ---------------------------------------------------------------------------

class TestModelSummaryNewlines:

    def test_model_summary_contains_actual_newlines(self, tmp_path):
        """T1: all write calls must use actual newline characters."""
        model_results = {
            "bayesian_r2": 0.6,
            "effect_sizes": {"log_ClusterVolume": 0.3, "P90": 0.1,
                             "RatioOverlap": 0.05, "DensityOverlap": 0.02},
            "loo": None,
            "nu_mean": 15.0,
            "nu_hdi": (5.0, 35.0),
            "outcome_stats": {"mean": -2.0, "std": 1.0},
        }
        summary_path = str(tmp_path / "model_summary.txt")
        generate_model_summary(model_results, summary_path)

        text = Path(summary_path).read_text()
        assert "\n" in text, "Model summary has no actual newlines"
        assert "\\n" not in text, "Model summary contains literal \\\\n (double-escaped)"

    def test_model_summary_structural_headers_on_separate_lines(self, tmp_path):
        model_results = {
            "bayesian_r2": 0.75,
            "effect_sizes": {"log_ClusterVolume": 0.4},
            "loo": None,
            "nu_mean": 8.0,
            "nu_hdi": (3.0, 20.0),
            "outcome_stats": {"mean": -1.5, "std": 0.8},
        }
        summary_path = str(tmp_path / "summary2.txt")
        generate_model_summary(model_results, summary_path)

        lines = Path(summary_path).read_text().splitlines()
        headers = [l.strip() for l in lines]
        assert "BAYESIAN CONNECTIVITY MODEL SUMMARY" in headers
        assert "MODEL FIT QUALITY" in headers
        assert "LIKELIHOOD ROBUSTNESS (Student-t)" in headers

    def test_nu_interpretation_heavy_tail(self, tmp_path):
        model_results = {
            "bayesian_r2": 0.5,
            "effect_sizes": {"log_ClusterVolume": 0.2},
            "loo": None,
            "nu_mean": 3.0,
            "nu_hdi": (1.5, 6.0),
            "outcome_stats": {},
        }
        summary_path = str(tmp_path / "summary_heavy.txt")
        generate_model_summary(model_results, summary_path)
        text = Path(summary_path).read_text()
        assert "heavy" in text.lower() or "Strong evidence" in text

    def test_nu_interpretation_normal_like(self, tmp_path):
        model_results = {
            "bayesian_r2": 0.8,
            "effect_sizes": {"log_ClusterVolume": 0.5},
            "loo": None,
            "nu_mean": 50.0,
            "nu_hdi": (20.0, 90.0),
            "outcome_stats": {},
        }
        summary_path = str(tmp_path / "summary_normal.txt")
        generate_model_summary(model_results, summary_path)
        text = Path(summary_path).read_text()
        assert "Normal" in text or "normal" in text


# ---------------------------------------------------------------------------
# run_connectivity_analysis — validation integration tests
# (Bayesian sampling hard-bypassed via monkeypatch)
# ---------------------------------------------------------------------------

def _make_minimal_cluster_img(tmp_path, shape, voxel_size, val=0.8,
                               name="cluster.nii.gz"):
    data = np.full(shape, val, dtype=np.float32)
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data)
    p = tmp_path / name
    _save_img(img, p)
    return p


def _make_minimal_patient_img(tmp_path, shape, voxel_size, val=0.1,
                               name="patient.nii.gz"):
    data = np.full(shape, val, dtype=np.float32)
    img = _make_img(shape=shape, voxel_size=voxel_size, data=data)
    p = tmp_path / name
    _save_img(img, p)
    return p


def _mock_run_bayesian_model(df):
    """Minimal stub returning the dict structure expected by generate_report."""
    import numpy as np
    import xarray as xr

    n = len(df)
    mu_data = np.ones((1, 10, n)) * 0.5
    mu_da = xr.DataArray(mu_data, dims=["chain", "draw", "mu_det_dim_0"])

    obs_data = np.ones((1, 10, n)) * 0.5
    obs_da = xr.DataArray(obs_data, dims=["chain", "draw", "obs_dim_0"])

    trace = type("FakeTrace", (), {
        "posterior": type("P", (), {
            "__getitem__": staticmethod(lambda k: mu_da),
            "beta_cv": xr.DataArray(np.array([0.1])),
            "beta_p90": xr.DataArray(np.array([0.1])),
            "beta_ro": xr.DataArray(np.array([0.1])),
            "beta_de": xr.DataArray(np.array([0.1])),
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
            "log_ClusterVolume", "P90", "RatioOverlap", "DensityOverlap",
        ]},
        "standardization_params": {},
        "nu_mean": 15.0,
        "nu_hdi": (5.0, 35.0),
        "outcome_stats": {"mean": -2.0, "std": 1.0},
    }


class TestRunConnectivityAnalysisValidation:

    def test_empty_overlap_skip_writes_diagnostic(self, tmp_path, monkeypatch):
        """
        When the preserved map has no overlap with the parcellation (patient
        disconnects everything), the script should skip the patient, write a
        diagnostic file, and NOT crash.
        """
        shape = (10, 10, 10)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=1.0)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        diag_files = list(out.glob("*_FAILED_diagnostic.txt"))
        assert len(diag_files) == 1, "Expected one diagnostic file"
        text = diag_files[0].read_text()
        assert "ANALYSIS FAILED" in text
        assert "POSSIBLE CAUSES" in text
        assert not list(out.glob("*_analysis_report.csv"))

    def test_empty_overlap_diagnostic_contains_input_info(self, tmp_path, monkeypatch):
        shape = (10, 10, 10)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=1.0,
                                            name="p1.nii.gz")
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        run_connectivity_analysis([str(patient)], [str(cluster)], str(parc), str(out))

        diag_files = list(out.glob("*_FAILED_diagnostic.txt"))
        text = diag_files[0].read_text()
        assert "(10, 10, 10)" in text
        assert "3" in text

    def test_multiple_patients_empty_and_valid(self, tmp_path, monkeypatch):
        """
        With two patients — one fully disconnected (empty overlap) and one
        with non-zero overlap — only the failing patient should produce a
        diagnostic file; the passing patient should produce a simplified report.
        """
        import xarray as xr

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)

        p_fail = tmp_path / "pat_fail.nii.gz"
        _save_img(_make_img(shape=shape, voxel_size=2.0,
                            data=np.ones(shape, dtype=np.float32)), p_fail)

        p_ok = tmp_path / "pat_ok.nii.gz"
        _save_img(_make_img(shape=shape, voxel_size=2.0,
                            data=np.full(shape, 0.1, dtype=np.float32)), p_ok)

        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None:
                Path(output_csv).write_text("ok\n"),
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
        but not abort.  Computation proceeds even though the result is spatially wrong.
        """
        shape = (10, 10, 10)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=5.0, val=0.8)
        out = tmp_path / "out"
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None:
                Path(output_csv).write_text("ok\n"),
        )
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
        assert "WARNING" in out_text

    def test_numpy_shape_error_wrapped_with_context(self, tmp_path, monkeypatch):
        """
        If numpy raises a broadcast error during preserved = cluster * (1 - patient),
        it should be caught and re-raised with context.
        """
        shape = (10, 10, 10)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap._check_image_space",
            lambda *args, **kwargs: {"issues": []},
        )

        original_load = nib.load
        call_count = [0]

        def patched_load(path):
            img = original_load(path)
            call_count[0] += 1
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


# ---------------------------------------------------------------------------
# Dual CSV output (T6)
# ---------------------------------------------------------------------------

class TestDualCsvOutput:
    """generate_report must write both a simplified and a full CSV."""

    def test_both_csv_files_created(self, tmp_path, monkeypatch):
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr("bcblib.tools.best_overlap.plt", _MockPlt())

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        simp_files = list(out.glob("*_analysis_report.csv"))
        full_files = list(out.glob("*_analysis_report_full.csv"))
        assert len(simp_files) == 1, "Simplified CSV not created"
        assert len(full_files) == 1, "Full CSV not created"

    def test_simplified_csv_has_expected_columns(self, tmp_path, monkeypatch):
        import pandas as pd
        from bcblib.tools.best_overlap import _SIMPLIFIED_COLS

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr("bcblib.tools.best_overlap.plt", _MockPlt())

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        simp_csv = list(out.glob("*_analysis_report.csv"))[0]
        df = pd.read_csv(simp_csv)
        for col in _SIMPLIFIED_COLS:
            assert col in df.columns, f"Expected column '{col}' missing from simplified CSV"

    def test_full_csv_has_all_metric_columns(self, tmp_path, monkeypatch):
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr("bcblib.tools.best_overlap.plt", _MockPlt())

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        full_csv = list(out.glob("*_analysis_report_full.csv"))[0]
        df = pd.read_csv(full_csv)
        for col in ["ROPE_Category", "ROPE_High", "CI_Width", "RelativeUncertainty",
                    "SumProb", "DensityOverlap", "ClusterVolume", "OverlapVolume"]:
            assert col in df.columns, f"Expected column '{col}' missing from full CSV"

    def test_rope_category_values_are_valid(self, tmp_path, monkeypatch):
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr("bcblib.tools.best_overlap.plt", _MockPlt())

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        simp_csv = list(out.glob("*_analysis_report.csv"))[0]
        df = pd.read_csv(simp_csv)
        valid_categories = {"Meaningful", "Negligible", "Inconclusive"}
        for cat in df["ROPE_Category"]:
            assert cat in valid_categories, f"Unexpected ROPE_Category value: {cat!r}"

    def test_simplified_csv_is_subset_of_full_csv_rows(self, tmp_path, monkeypatch):
        """Both CSVs should cover the same regions (same number of rows)."""
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.run_bayesian_model", _mock_run_bayesian_model
        )
        monkeypatch.setattr("bcblib.tools.best_overlap.plt", _MockPlt())

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out)
        )

        simp_df = pd.read_csv(list(out.glob("*_analysis_report.csv"))[0])
        full_df = pd.read_csv(list(out.glob("*_analysis_report_full.csv"))[0])
        assert len(simp_df) == len(full_df)


# ---------------------------------------------------------------------------
# Detection-limit filters (T7)
# ---------------------------------------------------------------------------

def _make_controlled_features_df(n_keep, n_remove):
    """Return a DataFrame with n_keep rows having high RatioOverlap/OverlapVolume
    and n_remove rows having very low values (will be caught by threshold filters)."""
    import pandas as pd

    rows = []
    for i in range(n_keep):
        rows.append({
            "Target_Region_ID": i + 1,
            "ClusterVolume": 100,
            "OverlapVolume": 90,
            "SumProb": 5.0,
            "P90": 0.3,
            "RatioOverlap": 0.9,
            "DensityOverlap": 0.055,
            "WeightedClusterContribution": 0.05,
        })
    for i in range(n_remove):
        rows.append({
            "Target_Region_ID": n_keep + i + 1,
            "ClusterVolume": 1000,
            "OverlapVolume": 1,
            "SumProb": 0.01,
            "P90": 0.01,
            "RatioOverlap": 0.001,
            "DensityOverlap": 0.01,
            "WeightedClusterContribution": 0.00001,
        })
    return pd.DataFrame(rows)


class TestDetectionLimitFilters:
    """T7: detection-limit filters applied with AND logic and proper warnings."""

    def _monkeypatch_features(self, monkeypatch, df_factory):
        """Patch compute_cluster_features to return df_factory() each call."""
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.compute_cluster_features",
            lambda preserved_map, parc_path: df_factory(),
        )

    def test_ratio_filter_removes_low_ratio_rows(self, tmp_path, monkeypatch):
        """min_ratio_overlap=0.01 removes rows with RatioOverlap=0.001."""
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        self._monkeypatch_features(monkeypatch,
                                   lambda: _make_controlled_features_df(n_keep=2, n_remove=1))

        seen_dfs = []

        def capture_model(df):
            seen_dfs.append(df.copy())
            return _mock_run_bayesian_model(df)

        monkeypatch.setattr("bcblib.tools.best_overlap.run_bayesian_model", capture_model)
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None: None,
        )

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out),
            min_ratio_overlap=0.01,
        )

        assert len(seen_dfs) == 1
        assert len(seen_dfs[0]) == 2, (
            f"Expected 2 rows after ratio filter, got {len(seen_dfs[0])}"
        )

    def test_voxel_filter_removes_low_overlap_rows(self, tmp_path, monkeypatch):
        """min_overlap_voxels=5 removes rows with OverlapVolume=1."""
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        self._monkeypatch_features(monkeypatch,
                                   lambda: _make_controlled_features_df(n_keep=2, n_remove=1))

        seen_dfs = []

        def capture_model(df):
            seen_dfs.append(df.copy())
            return _mock_run_bayesian_model(df)

        monkeypatch.setattr("bcblib.tools.best_overlap.run_bayesian_model", capture_model)
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None: None,
        )

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out),
            min_overlap_voxels=5,
        )

        assert len(seen_dfs) == 1
        assert len(seen_dfs[0]) == 2

    def test_and_logic_row_must_pass_both_filters(self, tmp_path, monkeypatch):
        """A pair failing either filter is excluded; AND logic, not OR."""
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        # Row 1 passes both; row 2 fails ratio; row 3 fails voxels
        def make_df():
            return pd.DataFrame([
                {"Target_Region_ID": 1, "ClusterVolume": 100, "OverlapVolume": 90,
                 "SumProb": 5.0, "P90": 0.3, "RatioOverlap": 0.9,
                 "DensityOverlap": 0.055, "WeightedClusterContribution": 0.05},
                {"Target_Region_ID": 2, "ClusterVolume": 10000, "OverlapVolume": 90,
                 "SumProb": 4.0, "P90": 0.25, "RatioOverlap": 0.009,
                 "DensityOverlap": 0.044, "WeightedClusterContribution": 0.0004},
                {"Target_Region_ID": 3, "ClusterVolume": 10, "OverlapVolume": 1,
                 "SumProb": 0.5, "P90": 0.5, "RatioOverlap": 0.1,
                 "DensityOverlap": 0.5, "WeightedClusterContribution": 0.05},
            ])

        monkeypatch.setattr(
            "bcblib.tools.best_overlap.compute_cluster_features",
            lambda *a, **kw: make_df(),
        )

        seen_dfs = []

        def capture_model(df):
            seen_dfs.append(df.copy())
            return _mock_run_bayesian_model(df)

        monkeypatch.setattr("bcblib.tools.best_overlap.run_bayesian_model", capture_model)
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None: None,
        )

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out),
            min_ratio_overlap=0.01,
            min_overlap_voxels=5,
        )

        assert len(seen_dfs) == 1
        assert len(seen_dfs[0]) == 1
        assert int(seen_dfs[0].iloc[0]["Target_Region_ID"]) == 1

    def test_warning_printed_when_above_20_percent_removed(
            self, tmp_path, monkeypatch, capsys):
        """If > 20 % of pairs are removed a WARNING mentioning '20' is printed."""
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        # 2 keep + 3 remove = 60% removed > 20%
        self._monkeypatch_features(monkeypatch,
                                   lambda: _make_controlled_features_df(n_keep=2, n_remove=3))

        monkeypatch.setattr("bcblib.tools.best_overlap.run_bayesian_model",
                            _mock_run_bayesian_model)
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None: None,
        )

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out),
            min_ratio_overlap=0.01,
        )

        out_text = capsys.readouterr().out
        assert "WARNING" in out_text
        assert "20" in out_text

    def test_no_filter_when_both_none(self, tmp_path, monkeypatch):
        """When both filters are None, all rows reach run_bayesian_model."""
        import pandas as pd

        shape = (12, 12, 12)
        parc = _make_parcellation(tmp_path, shape=shape, voxel_size=2.0, n_regions=3)
        cluster = _make_minimal_cluster_img(tmp_path, shape, voxel_size=2.0, val=0.8)
        patient = _make_minimal_patient_img(tmp_path, shape, voxel_size=2.0, val=0.1)
        out = tmp_path / "out"

        # 2 rows with low ratio that would be removed if filter were active
        self._monkeypatch_features(monkeypatch,
                                   lambda: _make_controlled_features_df(n_keep=2, n_remove=2))

        seen_dfs = []

        def capture_model(df):
            seen_dfs.append(df.copy())
            return _mock_run_bayesian_model(df)

        monkeypatch.setattr("bcblib.tools.best_overlap.run_bayesian_model", capture_model)
        monkeypatch.setattr(
            "bcblib.tools.best_overlap.generate_report",
            lambda df, model_results, output_csv, rope_high=None: None,
        )

        run_connectivity_analysis(
            [str(patient)], [str(cluster)], str(parc), str(out),
        )

        assert len(seen_dfs) == 1
        assert len(seen_dfs[0]) == 4, (
            f"Expected all 4 rows without filter, got {len(seen_dfs[0])}"
        )
