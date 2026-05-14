"""Tests for bcblib.tools.lesion_features."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data.astype(np.float32), affine)


def _save_nifti(path, data, affine=None):
    nib.save(_make_nifti(data, affine), str(path))


def _make_mni6_1mm():
    """Return a tiny NIfTI that pretends to be MNI6 1mm (shape 182×218×182)."""
    data = np.zeros((182, 218, 182), dtype=np.float32)
    data[90, 109, 90] = 1.0
    return _make_nifti(data)


def _make_mni6_2mm():
    """Return a tiny NIfTI at MNI6 2mm shape (91×109×91)."""
    data = np.zeros((91, 109, 91), dtype=np.float32)
    data[45, 54, 45] = 1.0
    return _make_nifti(data)


# ---------------------------------------------------------------------------
# T1 — BIDS utilities
# ---------------------------------------------------------------------------

class TestBidsUtils:

    def test_parse_entities_full(self):
        from bcblib.tools.lesion_features._bids import parse_bids_entities
        path = "sub-001_ses-01_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        e = parse_bids_entities(path)
        assert e["sub"] == "001"
        assert e["ses"] == "01"
        assert e["space"] == "MNI152NLin6Asym"
        assert e["res"] == "1"
        assert e["label"] == "lesion"
        assert e["suffix"] == "mask"
        assert e["extension"] == ".nii.gz"

    def test_parse_entities_minimal(self):
        from bcblib.tools.lesion_features._bids import parse_bids_entities
        path = "sub-002_label-lesion_mask.nii.gz"
        e = parse_bids_entities(path)
        assert e["sub"] == "002"
        assert e["ses"] is None
        assert e["res"] is None
        assert e["space"] is None

    def test_parse_entities_returns_none_for_missing(self):
        from bcblib.tools.lesion_features._bids import parse_bids_entities
        e = parse_bids_entities("sub-003_mask.nii.gz")
        assert e["ses"] is None
        assert e["space"] is None
        assert e["desc"] is None

    def test_build_lf_csv_path(self):
        from bcblib.tools.lesion_features._bids import build_lf_csv_path
        p = build_lf_csv_path("/out", "001", None, "MNI152NLin6Asym", "lesion", "JHU")
        assert p.name == "sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-JHU.csv"
        assert "sub-001" in str(p)
        assert "ses-" not in str(p)

    def test_build_lf_csv_path_with_ses(self):
        from bcblib.tools.lesion_features._bids import build_lf_csv_path
        p = build_lf_csv_path("/out", "001", "01", "MNI152NLin6Asym", "disconnectome", "AAL")
        assert "ses-01" in p.name
        assert "ses-01" in str(p.parent)

    def test_build_lf_tsv_path(self):
        from bcblib.tools.lesion_features._bids import build_lf_tsv_path
        p = build_lf_tsv_path("/out", "001", None, "MNI152NLin6Asym", "lesion")
        assert p.name == "sub-001_space-MNI152NLin6Asym_desc-lesion_mapstats.tsv"

    def test_build_lf_tsv_path_with_ses(self):
        from bcblib.tools.lesion_features._bids import build_lf_tsv_path
        p = build_lf_tsv_path("/out", "001", "02", "MNI152NLin6Asym", "disconnectome")
        assert "ses-02" in p.name

    def test_build_prep_path(self):
        from bcblib.tools.lesion_features._bids import build_prep_path
        p = build_prep_path("/prep", "001", None, "mask", "label-lesion")
        assert p.name == "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        assert p.parent.name == "anat"
        assert p.parent.parent.name == "sub-001"

    def test_build_prep_path_with_ses(self):
        from bcblib.tools.lesion_features._bids import build_prep_path
        p = build_prep_path("/prep", "001", "01", "mask", "label-lesion")
        assert "ses-01" in p.name
        assert p.parent.parent.name == "ses-01"

    def test_iter_bids_lesions_flat(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_bids_lesions
        anat = tmp_path / "sub-001" / "anat"
        anat.mkdir(parents=True)
        f = anat / "sub-001_space-MNI152NLin6Asym_label-lesion_mask.nii.gz"
        f.touch()
        results = list(iter_bids_lesions(tmp_path))
        assert len(results) == 1
        sub_id, ses_id, path = results[0]
        assert sub_id == "001"
        assert ses_id is None
        assert path == f

    def test_iter_bids_lesions_with_ses(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_bids_lesions
        anat = tmp_path / "sub-002" / "ses-01" / "anat"
        anat.mkdir(parents=True)
        f = anat / "sub-002_ses-01_label-lesion_mask.nii.gz"
        f.touch()
        results = list(iter_bids_lesions(tmp_path))
        assert len(results) == 1
        sub_id, ses_id, path = results[0]
        assert sub_id == "002"
        assert ses_id == "01"

    def test_iter_bids_lesions_multiple_subjects(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_bids_lesions
        for sub in ("001", "002", "003"):
            anat = tmp_path / f"sub-{sub}" / "anat"
            anat.mkdir(parents=True)
            (anat / f"sub-{sub}_label-lesion_mask.nii.gz").touch()
        results = list(iter_bids_lesions(tmp_path))
        assert len(results) == 3
        sub_ids = [r[0] for r in results]
        assert sorted(sub_ids) == ["001", "002", "003"]


# ---------------------------------------------------------------------------
# T2 — Space normalisation
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_extract_space_from_filename_known(self):
        from bcblib.tools.lesion_features._preprocess import extract_space_from_filename
        assert extract_space_from_filename(
            "sub-001_space-MNI152NLin6Asym_res-1_mask.nii.gz"
        ) == "MNI152NLin6Asym"
        assert extract_space_from_filename(
            "sub-001_space-MNI152NLin2009cAsym_mask.nii.gz"
        ) == "MNI152NLin2009cAsym"

    def test_extract_space_from_filename_none(self):
        from bcblib.tools.lesion_features._preprocess import extract_space_from_filename
        assert extract_space_from_filename("sub-001_label-lesion_mask.nii.gz") is None

    def test_extract_resolution_from_filename(self):
        from bcblib.tools.lesion_features._preprocess import extract_resolution_from_filename
        assert extract_resolution_from_filename(
            "sub-001_space-MNI152NLin6Asym_res-1_mask.nii.gz"
        ) == 1
        assert extract_resolution_from_filename(
            "sub-001_res-2_mask.nii.gz"
        ) == 2
        assert extract_resolution_from_filename("sub-001_mask.nii.gz") is None

    def test_detect_resolution_from_shape_1mm(self):
        from bcblib.tools.lesion_features._preprocess import detect_resolution_from_shape
        img = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))
        assert detect_resolution_from_shape(img) == 1
        img2 = _make_nifti(np.zeros((193, 229, 193), dtype=np.float32))
        assert detect_resolution_from_shape(img2) == 1

    def test_detect_resolution_from_shape_2mm(self):
        from bcblib.tools.lesion_features._preprocess import detect_resolution_from_shape
        img = _make_nifti(np.zeros((91, 109, 91), dtype=np.float32))
        assert detect_resolution_from_shape(img) == 2
        img2 = _make_nifti(np.zeros((97, 115, 97), dtype=np.float32))
        assert detect_resolution_from_shape(img2) == 2

    def test_detect_resolution_from_shape_unknown_raises(self):
        from bcblib.tools.lesion_features._preprocess import detect_resolution_from_shape
        img = _make_nifti(np.zeros((50, 60, 50), dtype=np.float32))
        with pytest.raises(ValueError, match="Unrecognised image shape"):
            detect_resolution_from_shape(img)

    def test_normalise_passthrough(self):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img = _make_mni6_1mm()
        result = normalise_lesion_to_mni6(img, "MNI152NLin6Asym", 1)
        assert result is img  # unchanged

    def test_normalise_resample_only(self):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img_2mm = _make_mni6_2mm()
        fake_ref = _make_mni6_1mm()
        fake_resampled = _make_mni6_1mm()

        with patch(
            "bcblib.tools.lesion_features._preprocess.resample_to_img",
            return_value=fake_resampled,
        ) as mock_rs, patch(
            "templateflow.api.get", return_value=["dummy_ref.nii.gz"]
        ), patch(
            "bcblib.tools.lesion_features._preprocess.nib.load", return_value=fake_ref
        ):
            result = normalise_lesion_to_mni6(img_2mm, "MNI152NLin6Asym", 2)

        mock_rs.assert_called_once()
        assert result is fake_resampled

    def test_normalise_warp_required(self):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img_2009c = _make_mni6_1mm()
        fake_warped = _make_mni6_1mm()

        with patch(
            "bcblib.tools.lesion_features._preprocess._apply_templateflow_warp",
            return_value=fake_warped,
        ) as mock_warp:
            result = normalise_lesion_to_mni6(img_2009c, "MNI152NLin2009cAsym", 1)

        mock_warp.assert_called_once()
        assert result is fake_warped

    def test_normalise_unsupported_space_raises(self):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img = _make_mni6_1mm()
        with pytest.raises(ValueError, match="Unsupported source space"):
            normalise_lesion_to_mni6(img, "SomeOtherSpace", 1)

    def test_normalise_load_from_path_string(self, tmp_path):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        p = tmp_path / "lesion.nii.gz"
        _save_nifti(p, np.zeros((182, 218, 182), dtype=np.float32))
        result = normalise_lesion_to_mni6(str(p), "MNI152NLin6Asym", 1)
        assert result is not None

    def test_normalise_source_res_none_detected(self, tmp_path):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))
        # source_res=None → should detect 1mm from shape, then pass through
        result = normalise_lesion_to_mni6(img, "MNI152NLin6Asym", None)
        assert result is img

    def test_normalise_mni2009c_warp_and_resample(self, tmp_path):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img_2009c = _make_nifti(np.zeros((91, 109, 91), dtype=np.float32))
        fake_warped = _make_nifti(np.zeros((91, 109, 91), dtype=np.float32))
        fake_ref = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))
        fake_resampled = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))

        with patch(
            "bcblib.tools.lesion_features._preprocess._apply_templateflow_warp",
            return_value=fake_warped,
        ), patch(
            "bcblib.tools.lesion_features._preprocess.resample_to_img",
            return_value=fake_resampled,
        ), patch(
            "templateflow.api.get", return_value=["dummy.nii.gz"]
        ), patch(
            "bcblib.tools.lesion_features._preprocess.nib.load", return_value=fake_ref
        ):
            result = normalise_lesion_to_mni6(img_2009c, "MNI152NLin2009cAsym", 2)

        assert result is fake_resampled

    def test_preprocess_one_saves_file(self, tmp_path):
        from bcblib.tools.lesion_features._preprocess import preprocess_one
        lesion = tmp_path / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        _save_nifti(lesion, np.zeros((182, 218, 182), dtype=np.float32))
        out = preprocess_one(lesion, tmp_path / "prep", "001")
        assert out.exists()
        assert "sub-001" in out.name
        assert "label-lesion" in out.name

    def test_preprocess_one_with_ses(self, tmp_path):
        from bcblib.tools.lesion_features._preprocess import preprocess_one
        lesion = tmp_path / "sub-001_ses-01_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        _save_nifti(lesion, np.zeros((182, 218, 182), dtype=np.float32))
        out = preprocess_one(lesion, tmp_path / "prep", "001", ses="01")
        assert "ses-01" in out.name


# ---------------------------------------------------------------------------
# T3 — Disconnectome runner
# ---------------------------------------------------------------------------

class TestDiscoRunner:

    def test_find_bcbtoolkit_default(self, tmp_path):
        from bcblib.tools.lesion_features._disco import find_bcbtoolkit
        fake_kit = tmp_path / "BCBToolKit"
        fake_kit.mkdir()
        (fake_kit / "run_disco.sh").touch()
        with patch(
            "bcblib.tools.lesion_features._disco.DEFAULT_BCBTOOLKIT",
            str(fake_kit),
        ):
            p = find_bcbtoolkit()
        assert p == fake_kit

    def test_find_bcbtoolkit_env_override(self, tmp_path):
        from bcblib.tools.lesion_features._disco import find_bcbtoolkit
        fake_kit = tmp_path / "custom_kit"
        fake_kit.mkdir()
        (fake_kit / "run_disco.sh").touch()
        with patch.dict("os.environ", {"BCBTOOLKIT_PATH": str(fake_kit)}):
            p = find_bcbtoolkit()
        assert p == fake_kit

    def test_find_bcbtoolkit_raises_if_missing(self, tmp_path):
        from bcblib.tools.lesion_features._disco import find_bcbtoolkit
        with patch(
            "bcblib.tools.lesion_features._disco.DEFAULT_BCBTOOLKIT",
            str(tmp_path / "nonexistent"),
        ), patch.dict("os.environ", {}, clear=True):
            with pytest.raises(FileNotFoundError, match="run_disco.sh not found"):
                find_bcbtoolkit()

    def test_predict_disco_output(self, tmp_path):
        from bcblib.tools.lesion_features._disco import predict_disco_output
        inp = "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        out = predict_disco_output(inp, tmp_path)
        assert out.name == "sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz"
        assert "_label-lesion_mask" not in out.name

    def test_run_disco_batch_calls_subprocess(self, tmp_path):
        from bcblib.tools.lesion_features._disco import run_disco_batch
        lesion_dir = tmp_path / "lesions"
        lesion_dir.mkdir()
        disco_dir = tmp_path / "disco"
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()

        (lesion_dir / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz").touch()

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            run_disco_batch(lesion_dir, disco_dir, fake_kit)

        args = mock_run.call_args[0][0]
        assert "-r" in args
        rename_val = args[args.index("-r") + 1]
        assert "_label-lesion_mask:_desc-disconnectome" == rename_val

    def test_run_disco_batch_raises_on_failure(self, tmp_path):
        from bcblib.tools.lesion_features._disco import run_disco_batch
        lesion_dir = tmp_path / "lesions"
        lesion_dir.mkdir()
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="run_disco.sh failed"):
                run_disco_batch(lesion_dir, tmp_path / "disco", fake_kit)

    def test_find_bcbtoolkit_path_hint(self, tmp_path):
        from bcblib.tools.lesion_features._disco import find_bcbtoolkit
        fake_kit = tmp_path / "hint_kit"
        fake_kit.mkdir()
        (fake_kit / "run_disco.sh").touch()
        p = find_bcbtoolkit(path_hint=str(fake_kit))
        assert p == fake_kit

    def test_run_disco_batch_with_ncores(self, tmp_path):
        from bcblib.tools.lesion_features._disco import run_disco_batch
        lesion_dir = tmp_path / "lesions"
        lesion_dir.mkdir()
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            run_disco_batch(lesion_dir, tmp_path / "disco", fake_kit, ncores=4)

        args = mock_run.call_args[0][0]
        assert "-n" in args
        assert "4" in args

    def test_collect_disco_outputs_finds_expected_files(self, tmp_path):
        from bcblib.tools.lesion_features._disco import collect_disco_outputs
        f1 = tmp_path / "sub-001_desc-disconnectome.nii.gz"
        f2 = tmp_path / "sub-002_desc-disconnectome.nii.gz"
        f1.touch()
        # f2 absent → warning
        expected = {"001": f1, "002": f2}
        with pytest.warns(RuntimeWarning, match="sub-002"):
            valid = collect_disco_outputs(tmp_path, expected)
        assert "001" in valid
        assert "002" not in valid


# ---------------------------------------------------------------------------
# T4 — Preprocessing pipeline orchestration
# ---------------------------------------------------------------------------

class TestPipelines:

    def test_preprocess_batch_creates_expected_files(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import preprocess_batch
        for sub in ("001", "002"):
            anat = tmp_path / "bids" / f"sub-{sub}" / "anat"
            anat.mkdir(parents=True)
            _save_nifti(
                anat / f"sub-{sub}_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
                np.zeros((182, 218, 182), dtype=np.float32),
            )
        prep = tmp_path / "prep"
        results = preprocess_batch(tmp_path / "bids", prep)
        assert len(results) == 2
        for path in results.values():
            assert path.exists()

    def test_preprocess_batch_skips_existing(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import preprocess_batch
        from bcblib.tools.lesion_features._bids import build_prep_path
        bids = tmp_path / "bids"
        anat = bids / "sub-001" / "anat"
        anat.mkdir(parents=True)
        _save_nifti(
            anat / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
            np.zeros((182, 218, 182), dtype=np.float32),
        )
        prep = tmp_path / "prep"
        # create the output in advance
        out = build_prep_path(prep, "001", None, "mask", "label-lesion")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        call_count = {"n": 0}
        orig = __import__(
            "bcblib.tools.lesion_features._preprocess", fromlist=["preprocess_one"]
        ).preprocess_one

        def spy(*a, **kw):
            call_count["n"] += 1
            return orig(*a, **kw)

        with patch(
            "bcblib.tools.lesion_features._pipeline.preprocess_one", side_effect=spy
        ):
            preprocess_batch(bids, prep, force=False)
        assert call_count["n"] == 0  # skipped

    # --- T5: feature extraction ---

    def _make_atlas_spec(self, tmp_path, name="test_atlas"):
        """Create a minimal label atlas and return an AtlasSpec for it."""
        from bcblib.tools.damage_profile import AtlasSpec
        arr = np.zeros((182, 218, 182), dtype=np.float32)
        arr[90, 109, 90] = 1.0  # single labelled region
        atlas_path = tmp_path / f"{name}.nii.gz"
        _save_nifti(atlas_path, arr)
        return AtlasSpec(source=str(atlas_path), name=name)

    def test_extract_features_one_writes_csvs(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec = self._make_atlas_spec(tmp_path)
        # lesion overlapping the atlas region
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        _save_nifti(lesion_path, lesion)

        written = extract_features_one("001", None, lesion_path, None, [spec], tmp_path / "out")

        assert f"lesion_{spec.name}_csv" in written
        assert written[f"lesion_{spec.name}_csv"].exists()

    def test_extract_features_one_writes_tsvs(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec = self._make_atlas_spec(tmp_path)
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        _save_nifti(lesion_path, lesion)

        written = extract_features_one("001", None, lesion_path, None, [spec], tmp_path / "out")

        assert "lesion_mapstats_tsv" in written
        assert written["lesion_mapstats_tsv"].exists()

    def test_extract_features_one_with_ses(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec = self._make_atlas_spec(tmp_path)
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_ses-01_label-lesion_mask.nii.gz"
        _save_nifti(lesion_path, lesion)

        written = extract_features_one("001", "01", lesion_path, None, [spec], tmp_path / "out")

        csv_path = written[f"lesion_{spec.name}_csv"]
        assert "ses-01" in csv_path.name
        assert "ses-01" in str(csv_path.parent)

    def test_extract_features_one_disco_written(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec = self._make_atlas_spec(tmp_path)
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        disco_path = tmp_path / "sub-001_desc-disconnectome.nii.gz"
        _save_nifti(lesion_path, lesion)
        _save_nifti(disco_path, lesion)  # same non-zero pattern

        written = extract_features_one("001", None, lesion_path, disco_path, [spec], tmp_path / "out")

        assert f"disconnectome_{spec.name}_csv" in written
        assert written[f"disconnectome_{spec.name}_csv"].exists()
        assert "disconnectome_mapstats_tsv" in written

    def test_extract_features_batch_all_subjects(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        lesion_data = np.zeros((182, 218, 182), dtype=np.float32)
        lesion_data[90, 109, 90] = 1.0
        prep = tmp_path / "prep"

        for sub in ("001", "002"):
            anat = prep / f"sub-{sub}" / "anat"
            anat.mkdir(parents=True)
            _save_nifti(
                anat / f"sub-{sub}_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
                lesion_data,
            )
            _save_nifti(
                anat / f"sub-{sub}_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz",
                lesion_data,
            )

        results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 2
        for sub_key in ("001", "002"):
            assert sub_key in results
            assert results[sub_key]  # non-empty dict of paths

    def test_extract_features_batch_with_ses(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        lesion_data = np.zeros((182, 218, 182), dtype=np.float32)
        lesion_data[90, 109, 90] = 1.0
        prep = tmp_path / "prep"

        for sub, ses in (("001", "01"), ("002", "02")):
            anat = prep / f"sub-{sub}" / f"ses-{ses}" / "anat"
            anat.mkdir(parents=True)
            _save_nifti(
                anat / f"sub-{sub}_ses-{ses}_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
                lesion_data,
            )
            _save_nifti(
                anat / f"sub-{sub}_ses-{ses}_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz",
                lesion_data,
            )

        results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 2
        assert "001_ses-01" in results
        assert "002_ses-02" in results

    def test_extract_features_batch_no_lesion_in_anat(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        prep = tmp_path / "prep"
        anat = prep / "sub-001" / "anat"
        anat.mkdir(parents=True)
        # no lesion file — should silently skip

        results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 0

    def test_extract_features_batch_missing_disco_warns(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        lesion_data = np.zeros((182, 218, 182), dtype=np.float32)
        lesion_data[90, 109, 90] = 1.0
        prep = tmp_path / "prep"

        anat = prep / "sub-001" / "anat"
        anat.mkdir(parents=True)
        _save_nifti(
            anat / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
            lesion_data,
        )
        # no disconnectome file → should warn and skip

        with pytest.warns(RuntimeWarning, match="No disconnectome"):
            results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 0

    def test_preprocess_batch_handles_missing_space_in_filename(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import preprocess_batch
        bids = tmp_path / "bids"
        anat = bids / "sub-001" / "anat"
        anat.mkdir(parents=True)
        # no _space- entity in filename; shape-based detection falls back to 1mm
        _save_nifti(
            anat / "sub-001_label-lesion_mask.nii.gz",
            np.zeros((182, 218, 182), dtype=np.float32),
        )
        prep = tmp_path / "prep"
        results = preprocess_batch(bids, prep)
        assert "001" in results
        assert results["001"].exists()


# ---------------------------------------------------------------------------
# T7 — CLIs
# ---------------------------------------------------------------------------

class TestCLI:

    def test_lf_preprocess_help(self):
        from bcblib.scripts.run_lf_preprocess import _build_parser
        p = _build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--help"])
        assert exc.value.code == 0

    def test_lf_preprocess_missing_required_arg(self):
        from bcblib.scripts.run_lf_preprocess import main
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0

    def test_lesion_features_help(self):
        from bcblib.scripts.run_lesion_features import _build_parser
        p = _build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--help"])
        assert exc.value.code == 0

    def test_lesion_features_missing_required_arg(self):
        from bcblib.scripts.run_lesion_features import main
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0

    def test_lf_preprocess_dry_run(self, tmp_path):
        from bcblib.scripts.run_lf_preprocess import main
        bids = tmp_path / "bids"
        anat = bids / "sub-001" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-001_label-lesion_mask.nii.gz").touch()
        main(["--bids-dir", str(bids), "--output-dir", str(tmp_path / "prep"), "--dry-run"])

    def test_lesion_features_no_atlases_exits(self, tmp_path):
        from bcblib.scripts.run_lesion_features import main
        prep = tmp_path / "prep"
        prep.mkdir()
        with pytest.raises(SystemExit) as exc:
            main(["--prep-dir", str(prep)])
        assert exc.value.code != 0
