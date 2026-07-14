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

    def test_build_lf_csv_path_with_lesion_desc(self):
        from bcblib.tools.lesion_features._bids import build_lf_csv_path
        p = build_lf_csv_path("/out", "001", None, "MNI152NLin6Asym", "lesion", "aal",
                               lesion_desc="core")
        assert p.name == "sub-001_space-MNI152NLin6Asym_LF-lesion-core_atlas-aal.csv"

    def test_build_lf_tsv_path(self):
        from bcblib.tools.lesion_features._bids import build_lf_tsv_path
        p = build_lf_tsv_path("/out", "001", None, "MNI152NLin6Asym", "lesion")
        assert p.name == "sub-001_space-MNI152NLin6Asym_desc-lesion_mapstats.tsv"

    def test_build_lf_tsv_path_with_ses(self):
        from bcblib.tools.lesion_features._bids import build_lf_tsv_path
        p = build_lf_tsv_path("/out", "001", "02", "MNI152NLin6Asym", "disconnectome")
        assert "ses-02" in p.name

    def test_build_lf_tsv_path_with_lesion_desc(self):
        from bcblib.tools.lesion_features._bids import build_lf_tsv_path
        p = build_lf_tsv_path("/out", "001", None, "MNI152NLin6Asym", "lesion",
                               lesion_desc="edema")
        assert p.name == "sub-001_space-MNI152NLin6Asym_desc-edema-lesion_mapstats.tsv"

    def test_build_prep_path(self):
        from bcblib.tools.lesion_features._bids import build_prep_path
        p = build_prep_path("/prep", "001", None, "mask", "label-lesion")
        assert p.name == "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
        assert p.parent.name == "lesion"
        assert p.parent.parent.name == "sub-001"

    def test_build_prep_path_with_ses(self):
        from bcblib.tools.lesion_features._bids import build_prep_path
        p = build_prep_path("/prep", "001", "01", "mask", "label-lesion")
        assert "ses-01" in p.name
        assert p.parent.parent.name == "ses-01"

    def test_build_prep_path_with_lesion_desc(self):
        from bcblib.tools.lesion_features._bids import build_prep_path
        p = build_prep_path("/prep", "001", None, "mask", "label-lesion", lesion_desc="core")
        assert p.name == "sub-001_space-MNI152NLin6Asym_res-1_desc-core_label-lesion_mask.nii.gz"

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

    def test_iter_bids_lesions_custom_suffix(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_bids_lesions
        anat = tmp_path / "sub-001" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-001_space-MNI_label-tumor_mask.nii.gz").touch()
        results = list(iter_bids_lesions(tmp_path, suffix="*_label-tumor_mask.nii.gz"))
        assert len(results) == 1

    def test_iter_bids_lesions_multiple_desc(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_bids_lesions
        anat = tmp_path / "sub-001" / "anat"
        anat.mkdir(parents=True)
        for desc in ("core", "edema", "necrosis"):
            (anat / f"sub-001_space-MNI_desc-{desc}_label-lesion_mask.nii.gz").touch()
        results = list(iter_bids_lesions(tmp_path))
        assert len(results) == 3

    def test_iter_flat_lesions_single_subject(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_flat_lesions
        sub_dir = tmp_path / "sub-010"
        sub_dir.mkdir(parents=True)
        f = sub_dir / "mni_sub-010_lesion_mask.nii.gz"
        f.touch()
        results = list(iter_flat_lesions(tmp_path))
        assert len(results) == 1
        sub_id, ses_id, path = results[0]
        assert sub_id == "010"
        assert ses_id is None
        assert path == f

    def test_iter_flat_lesions_multiple_subjects(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_flat_lesions
        for sub in ("001", "002", "003"):
            d = tmp_path / f"sub-{sub}"
            d.mkdir()
            (d / f"mni_sub-{sub}_lesion_mask.nii.gz").touch()
        results = list(iter_flat_lesions(tmp_path))
        assert len(results) == 3
        assert sorted(r[0] for r in results) == ["001", "002", "003"]

    def test_iter_flat_lesions_ignores_anat_subdir(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_flat_lesions
        # File in anat/ should NOT be found by the flat iterator
        anat = tmp_path / "sub-001" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-001_label-lesion_mask.nii.gz").touch()
        results = list(iter_flat_lesions(tmp_path))
        assert len(results) == 0

    def test_iter_flat_lesions_custom_suffix(self, tmp_path):
        from bcblib.tools.lesion_features._bids import iter_flat_lesions
        sub_dir = tmp_path / "sub-001"
        sub_dir.mkdir()
        (sub_dir / "mni_sub-001_lesion_mask.nii.gz").touch()
        (sub_dir / "mni_sub-001_other.nii.gz").touch()
        results = list(iter_flat_lesions(tmp_path, suffix="*_lesion_mask.nii.gz"))
        assert len(results) == 1
        assert results[0][2].name == "mni_sub-001_lesion_mask.nii.gz"

    def test_preprocess_batch_flat_mode(self, tmp_path):
        import nibabel as nib
        import numpy as np
        from unittest.mock import patch
        from bcblib.tools.lesion_features._pipeline import preprocess_batch

        sub_dir = tmp_path / "input" / "sub-042"
        sub_dir.mkdir(parents=True)
        img = nib.Nifti1Image(np.zeros((182, 218, 182), dtype=np.uint8), np.eye(4))
        lesion = sub_dir / "mni_sub-042_lesion_mask.nii.gz"
        nib.save(img, str(lesion))

        prep = tmp_path / "prep"
        with patch(
            "bcblib.tools.lesion_features._preprocess.normalise_lesion_to_mni6",
            return_value=img,
        ):
            results = preprocess_batch(
                tmp_path / "input", prep,
                suffix="*_lesion_mask.nii.gz", flat=True,
            )

        assert "042" in results
        out = results["042"]
        assert out.exists()
        assert "sub-042" in str(out)
        assert "lesion" in str(out)

    def test_predict_disco_output_plain(self):
        from bcblib.tools.lesion_features._disco import predict_disco_output
        p = predict_disco_output(
            "sub-001_space-MNI_res-1_label-lesion_mask.nii.gz", "/out"
        )
        assert p.name == "sub-001_space-MNI_res-1_desc-disconnectome.nii.gz"

    def test_predict_disco_output_with_desc(self):
        from bcblib.tools.lesion_features._disco import predict_disco_output
        p = predict_disco_output(
            "sub-001_space-MNI_res-1_desc-core_label-lesion_mask.nii.gz", "/out"
        )
        assert p.name == "sub-001_space-MNI_res-1_desc-core-disconnectome.nii.gz"


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
        # SPM12 MNI152 1 mm template (slightly smaller FOV than FSL)
        img3 = _make_nifti(np.zeros((181, 217, 181), dtype=np.float32))
        assert detect_resolution_from_shape(img3) == 1

    def test_normalise_spm_shape_resampled_to_canonical(self):
        """SPM (181,217,181) lesion must be padded to FSL canonical (182,218,182)."""
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img = _make_nifti(np.zeros((181, 217, 181), dtype=np.uint8))
        out = normalise_lesion_to_mni6(img, source_space=None, source_res=1)
        assert out.shape[:3] == (182, 218, 182)

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
        fake_resampled = _make_mni6_1mm()

        with patch(
            "bcblib.tools.lesion_features._preprocess.resample_img",
            return_value=fake_resampled,
        ) as mock_rs:
            result = normalise_lesion_to_mni6(img_2mm, "MNI152NLin6Asym", 2)

        mock_rs.assert_called_once()
        assert result is fake_resampled

    def test_normalise_warp_required(self):
        from bcblib.tools.lesion_features._preprocess import normalise_lesion_to_mni6
        img_2009c = _make_mni6_1mm()
        fake_warped = _make_mni6_1mm()  # same orientation as input → no reorientation

        with patch(
            "bcblib.tools.lesion_features._preprocess.warp_binary_mask",
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
        # fake_warped has same orientation as input (identity) → no reorientation step
        fake_warped = _make_nifti(np.zeros((182, 218, 182), dtype=np.float32))

        with patch(
            "bcblib.tools.lesion_features._preprocess.warp_binary_mask",
            return_value=fake_warped,
        ):
            result = normalise_lesion_to_mni6(img_2009c, "MNI152NLin2009cAsym", 2)

        assert result is fake_warped

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

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            run_disco_batch(lesion_dir, disco_dir, fake_kit)

        args = mock_popen.call_args[0][0]
        assert "-r" in args
        rename_val = args[args.index("-r") + 1]
        assert "_label-lesion_mask:_desc-disconnectome" == rename_val

    def test_run_disco_batch_raises_on_failure(self, tmp_path):
        from bcblib.tools.lesion_features._disco import run_disco_batch
        lesion_dir = tmp_path / "lesions"
        lesion_dir.mkdir()
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1
        with patch("subprocess.Popen", return_value=mock_proc):
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

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            run_disco_batch(lesion_dir, tmp_path / "disco", fake_kit, ncores=4)

        args = mock_popen.call_args[0][0]
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
# T3b — Private TDI hook
# ---------------------------------------------------------------------------

class TestTdi:

    def _make_tdi_dir(self, tmp_path):
        tdi_dir = tmp_path / "tdi"
        tdi_dir.mkdir()
        (tdi_dir / "tdi.py").write_text(
            "def calculate_tdi(atlas_path, mask_path):\n"
            "    return 0.42\n"
        )
        _save_nifti(tdi_dir / "tdi_map_1mm.nii", np.zeros((4, 4, 4)))
        return tdi_dir

    def test_find_tdi_dir_default(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import find_tdi_dir
        fake_dir = self._make_tdi_dir(tmp_path)
        with patch("bcblib.tools.lesion_features._tdi.DEFAULT_TDI_DIR", str(fake_dir)):
            p = find_tdi_dir()
        assert p == fake_dir

    def test_find_tdi_dir_env_override(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import find_tdi_dir
        fake_dir = self._make_tdi_dir(tmp_path)
        with patch.dict("os.environ", {"TDI_DIR": str(fake_dir)}):
            p = find_tdi_dir()
        assert p == fake_dir

    def test_find_tdi_dir_path_hint(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import find_tdi_dir
        fake_dir = self._make_tdi_dir(tmp_path)
        p = find_tdi_dir(path_hint=str(fake_dir))
        assert p == fake_dir

    def test_find_tdi_dir_warns_if_missing(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import find_tdi_dir
        with patch(
            "bcblib.tools.lesion_features._tdi.DEFAULT_TDI_DIR",
            str(tmp_path / "nonexistent"),
        ), patch.dict("os.environ", {}, clear=True):
            with pytest.warns(RuntimeWarning, match="TDI script/atlas not found"):
                p = find_tdi_dir()
        assert p is None

    def test_load_tdi_function_returns_none_if_missing(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import load_tdi_function
        with patch(
            "bcblib.tools.lesion_features._tdi.DEFAULT_TDI_DIR",
            str(tmp_path / "nonexistent"),
        ), patch.dict("os.environ", {}, clear=True):
            with pytest.warns(RuntimeWarning):
                fn = load_tdi_function()
        assert fn is None

    def test_load_tdi_function_calls_calculate_tdi(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import load_tdi_function
        fake_dir = self._make_tdi_dir(tmp_path)
        lesion_path = tmp_path / "lesion.nii.gz"
        _save_nifti(lesion_path, np.zeros((4, 4, 4)))  # same orientation as atlas

        fn = load_tdi_function(path_hint=str(fake_dir))
        assert fn(lesion_path) == 0.42

    def test_load_tdi_function_reorients_mismatched_lesion(self, tmp_path):
        from bcblib.tools.lesion_features._tdi import load_tdi_function
        fake_dir = self._make_tdi_dir(tmp_path)
        # flip the x axis relative to the atlas (mirrors the real FSL-vs-
        # TemplateFlow orientation mismatch) — calculate_tdi ignores the data
        # and returns 0.42 regardless, so this only checks no error is raised.
        flipped_affine = np.eye(4)
        flipped_affine[0, 0] = -1
        lesion_path = tmp_path / "lesion_flipped.nii.gz"
        _save_nifti(lesion_path, np.zeros((4, 4, 4)), affine=flipped_affine)

        fn = load_tdi_function(path_hint=str(fake_dir))
        assert fn(lesion_path) == 0.42


# ---------------------------------------------------------------------------
# T4 — Streamline ratio
# ---------------------------------------------------------------------------

class TestStreamlineRatio:

    def _make_trk_dir(self, tmp_path):
        trk_dir = tmp_path / "trk"
        trk_dir.mkdir()
        (trk_dir / "AF_L.trk").touch()
        (trk_dir / "AF_R.trk").touch()
        return trk_dir

    def _make_atlas_spec(self, tmp_path, name="yeh_hcp1065"):
        from bcblib.tools.damage_profile import AtlasSpec
        arr = np.zeros((182, 218, 182), dtype=np.float32)
        arr[90, 109, 90] = 1.0
        atlas_path = tmp_path / f"{name}.nii.gz"
        _save_nifti(atlas_path, arr)
        return AtlasSpec(source=str(atlas_path), name=name)

    def test_returns_none_without_dipy(self, tmp_path):
        from bcblib.tools.lesion_features._streamline import load_streamline_ratio_function
        trk_dir = self._make_trk_dir(tmp_path)
        with patch("bcblib.tools.lesion_features._streamline._HAS_DIPY", False):
            with pytest.warns(RuntimeWarning, match="dipy"):
                fn = load_streamline_ratio_function(trk_dir)
        assert fn is None

    def test_returns_none_if_no_trk_files(self, tmp_path):
        from bcblib.tools.lesion_features._streamline import load_streamline_ratio_function
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("bcblib.tools.lesion_features._streamline._HAS_DIPY", True):
            with pytest.warns(RuntimeWarning, match="No .trk files"):
                fn = load_streamline_ratio_function(empty_dir)
        assert fn is None

    def test_returns_callable_when_ready(self, tmp_path):
        from bcblib.tools.lesion_features._streamline import load_streamline_ratio_function
        trk_dir = self._make_trk_dir(tmp_path)
        with patch("bcblib.tools.lesion_features._streamline._HAS_DIPY", True):
            fn = load_streamline_ratio_function(trk_dir)
        assert callable(fn)

    def test_extract_features_one_adds_streamline_ratio_to_yeh_lesion_csv(self, tmp_path):
        import pandas as pd
        from bcblib.tools.lesion_features._pipeline import extract_features_one

        spec = self._make_atlas_spec(tmp_path)
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        disco_path = tmp_path / "sub-001_desc-disconnectome.nii.gz"
        _save_nifti(lesion_path, lesion)
        _save_nifti(disco_path, lesion)

        mock_ratio = pd.DataFrame({"region_name": ["some_tract"], "streamline_ratio": [0.25]})
        streamline_fn = lambda p: mock_ratio  # noqa: E731

        written = extract_features_one(
            "001", None, lesion_path, disco_path, [spec], tmp_path / "out",
            streamline_fn=streamline_fn,
        )

        lesion_csv = pd.read_csv(written["lesion_yeh_hcp1065_csv"])
        assert "streamline_ratio" in lesion_csv.columns

        disco_csv = pd.read_csv(written["disconnectome_yeh_hcp1065_csv"])
        assert "streamline_ratio" not in disco_csv.columns

    def test_extract_features_one_no_streamline_ratio_for_other_atlas(self, tmp_path):
        import pandas as pd
        from bcblib.tools.lesion_features._pipeline import extract_features_one

        spec = self._make_atlas_spec(tmp_path, name="other_atlas")
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        _save_nifti(lesion_path, lesion)

        mock_ratio = pd.DataFrame({"region_name": ["t"], "streamline_ratio": [0.1]})
        streamline_fn = lambda p: mock_ratio  # noqa: E731

        written = extract_features_one(
            "001", None, lesion_path, None, [spec], tmp_path / "out",
            streamline_fn=streamline_fn,
        )

        csv = pd.read_csv(written["lesion_other_atlas_csv"])
        assert "streamline_ratio" not in csv.columns


# ---------------------------------------------------------------------------
# T5 — Preprocessing pipeline orchestration
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

    def test_extract_features_one_adds_tdi_to_lesion_tsv_only(self, tmp_path):
        import pandas as pd
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec = self._make_atlas_spec(tmp_path)
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        disco_path = tmp_path / "sub-001_desc-disconnectome.nii.gz"
        _save_nifti(lesion_path, lesion)
        _save_nifti(disco_path, lesion)

        tdi_fn = lambda p: 0.42  # noqa: E731

        written = extract_features_one(
            "001", None, lesion_path, disco_path, [spec], tmp_path / "out", tdi_fn=tdi_fn,
        )

        lesion_tsv = pd.read_csv(written["lesion_mapstats_tsv"], sep="\t")
        assert lesion_tsv["tdi"].iloc[0] == 0.42

        disco_tsv = pd.read_csv(written["disconnectome_mapstats_tsv"], sep="\t")
        assert "tdi" not in disco_tsv.columns

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
            lesion_dir = prep / f"sub-{sub}" / "lesion"
            lesion_dir.mkdir(parents=True)
            _save_nifti(
                lesion_dir / f"sub-{sub}_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
                lesion_data,
            )
            _save_nifti(
                lesion_dir / f"sub-{sub}_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz",
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
            lesion_dir = prep / f"sub-{sub}" / f"ses-{ses}" / "lesion"
            lesion_dir.mkdir(parents=True)
            _save_nifti(
                lesion_dir / f"sub-{sub}_ses-{ses}_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
                lesion_data,
            )
            _save_nifti(
                lesion_dir / f"sub-{sub}_ses-{ses}_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz",
                lesion_data,
            )

        results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 2
        assert "001_ses-01" in results
        assert "002_ses-02" in results

    def test_extract_features_batch_no_lesion_in_lesion_dir(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        prep = tmp_path / "prep"
        lesion_dir = prep / "sub-001" / "lesion"
        lesion_dir.mkdir(parents=True)
        # no lesion file — should silently skip

        results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 0

    def test_extract_features_batch_missing_disco_warns(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        lesion_data = np.zeros((182, 218, 182), dtype=np.float32)
        lesion_data[90, 109, 90] = 1.0
        prep = tmp_path / "prep"

        lesion_dir = prep / "sub-001" / "lesion"
        lesion_dir.mkdir(parents=True)
        _save_nifti(
            lesion_dir / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
            lesion_data,
        )
        # no disconnectome file → should warn and skip

        with pytest.warns(RuntimeWarning, match="No disconnectome"):
            results = extract_features_batch(prep, [spec], tmp_path / "out")
        assert len(results) == 0

    def test_extract_features_one_skips_all_when_complete(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec = self._make_atlas_spec(tmp_path)
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        _save_nifti(lesion_path, lesion)
        out = tmp_path / "out"

        written1 = extract_features_one("001", None, lesion_path, None, [spec], out, force=True)
        assert written1

        written2 = extract_features_one("001", None, lesion_path, None, [spec], out, force=False)
        assert not written2  # everything already on disk — nothing written

    def test_extract_features_one_writes_only_missing_atlas(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_one
        spec1 = self._make_atlas_spec(tmp_path, name="atlas1")
        spec2 = self._make_atlas_spec(tmp_path, name="atlas2")
        lesion = np.zeros((182, 218, 182), dtype=np.float32)
        lesion[90, 109, 90] = 1.0
        lesion_path = tmp_path / "sub-001_label-lesion_mask.nii.gz"
        _save_nifti(lesion_path, lesion)
        out = tmp_path / "out"

        # First run: only spec1
        extract_features_one("001", None, lesion_path, None, [spec1], out, force=True)

        # Second run: spec1 + spec2 with skip — should write only spec2
        written = extract_features_one("001", None, lesion_path, None, [spec1, spec2], out, force=False)
        assert "lesion_atlas2_csv" in written
        assert "lesion_atlas1_csv" not in written

    def test_extract_features_batch_skips_complete_subjects(self, tmp_path):
        from bcblib.tools.lesion_features._pipeline import extract_features_batch
        spec = self._make_atlas_spec(tmp_path)
        lesion_data = np.zeros((182, 218, 182), dtype=np.float32)
        lesion_data[90, 109, 90] = 1.0
        prep = tmp_path / "prep"
        lesion_dir = prep / "sub-001" / "lesion"
        lesion_dir.mkdir(parents=True)
        _save_nifti(
            lesion_dir / "sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz",
            lesion_data,
        )
        _save_nifti(
            lesion_dir / "sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz",
            lesion_data,
        )
        out = tmp_path / "out"

        results1 = extract_features_batch(prep, [spec], out, force=True)
        assert "001" in results1

        results2 = extract_features_batch(prep, [spec], out, force=False)
        assert len(results2) == 0  # fully done — skipped

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


class TestEbrainsAtlasSpecs:

    def test_ebrains_includes_rojkova_and_yeh(self):
        from bcblib.tools.lesion_features._constants import EBRAINS_ATLAS_SPECS
        assert "rojkova" in EBRAINS_ATLAS_SPECS
        assert "yeh_hcp1065" in EBRAINS_ATLAS_SPECS

    def test_ebrains_all_keys_are_registered_presets(self):
        from bcblib.tools.lesion_features._constants import EBRAINS_ATLAS_SPECS
        from bcblib.tools.damage_profile._atlas_manager import PRESET_ATLASES
        for key in EBRAINS_ATLAS_SPECS:
            assert key in PRESET_ATLASES, f"{key!r} not in PRESET_ATLASES"
