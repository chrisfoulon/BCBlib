"""Tests for bcblib.scripts.run_lf_preprocess's disconnectome engine selection."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from bcblib.scripts.run_lf_preprocess import _select_disco_engine


def _args(**overrides):
    defaults = dict(
        engine="auto",
        bcbtoolkit=None,
        disco2_index=None,
        ncores=None,
        tracks_dir=None,
        tmpdir=None,
        skip_existing=False,
        fiber_class=None,
        out_voxel_size=None,
        len_min=None,
        len_max=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestSelectDiscoEngine:

    def test_auto_prefers_disco2_when_ready(self, tmp_path):
        with patch(
            "bcblib.tools.lesion_features._disco.disco2_ready",
            return_value=tmp_path,
        ):
            engine, runner = _select_disco_engine(_args(engine="auto"))
        assert engine == "disco2"
        assert runner is not None

    def test_auto_falls_back_to_bcbtoolkit_when_disco2_not_ready(self, tmp_path):
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()
        (fake_kit / "run_disco.sh").touch()
        with patch(
            "bcblib.tools.lesion_features._disco.disco2_ready",
            return_value=None,
        ), patch(
            "bcblib.tools.lesion_features._disco.find_bcbtoolkit",
            return_value=fake_kit,
        ):
            engine, runner = _select_disco_engine(_args(engine="auto"))
        assert engine == "bcbtoolkit"
        assert runner is not None

    def test_auto_returns_none_when_neither_available(self):
        with patch(
            "bcblib.tools.lesion_features._disco.disco2_ready",
            return_value=None,
        ), patch(
            "bcblib.tools.lesion_features._disco.find_bcbtoolkit",
            side_effect=FileNotFoundError("BCBToolKit run_disco.sh not found."),
        ):
            engine, runner = _select_disco_engine(_args(engine="auto"))
        assert (engine, runner) == (None, None)

    def test_bcbtoolkit_forced_kit_missing_returns_none(self):
        with patch(
            "bcblib.tools.lesion_features._disco.find_bcbtoolkit",
            side_effect=FileNotFoundError("BCBToolKit run_disco.sh not found."),
        ):
            engine, runner = _select_disco_engine(_args(engine="bcbtoolkit"))
        assert (engine, runner) == (None, None)

    def test_bcbtoolkit_forced_kit_found(self, tmp_path):
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()
        with patch(
            "bcblib.tools.lesion_features._disco.find_bcbtoolkit",
            return_value=fake_kit,
        ):
            engine, runner = _select_disco_engine(_args(engine="bcbtoolkit"))
        assert engine == "bcbtoolkit"
        assert runner is not None

    def test_disco2_forced_and_ready(self, tmp_path):
        with patch(
            "bcblib.tools.lesion_features._disco.require_disco2",
            return_value=tmp_path,
        ):
            engine, runner = _select_disco_engine(_args(engine="disco2"))
        assert engine == "disco2"
        assert runner is not None

    def test_disco2_forced_and_not_ready_hard_fails(self):
        with patch(
            "bcblib.tools.lesion_features._disco.require_disco2",
            side_effect=ModuleNotFoundError("disconnectome2 is not installed"),
        ):
            with pytest.raises(SystemExit):
                _select_disco_engine(_args(engine="disco2"))

    # -- Phase 2: fiber_class/out_voxel_size/len_min/len_max wiring ---------

    def test_disco2_forced_threads_phase2_flags_into_runner(self, tmp_path):
        with patch(
            "bcblib.tools.lesion_features._disco.require_disco2",
            return_value=tmp_path,
        ):
            engine, runner = _select_disco_engine(_args(
                engine="disco2", fiber_class="association",
                out_voxel_size=2.0, len_min=20.0, len_max=150.0,
            ))
        assert engine == "disco2"
        assert runner.keywords["fiber_class"] == "association"
        assert runner.keywords["out_voxel_size"] == 2.0
        assert runner.keywords["len_min"] == 20.0
        assert runner.keywords["len_max"] == 150.0

    def test_bcbtoolkit_forced_warns_when_phase2_flags_set(self, tmp_path, capsys):
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()
        with patch(
            "bcblib.tools.lesion_features._disco.find_bcbtoolkit",
            return_value=fake_kit,
        ):
            engine, runner = _select_disco_engine(
                _args(engine="bcbtoolkit", fiber_class="association")
            )
        assert engine == "bcbtoolkit"
        assert runner is not None
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "--fiber-class" in err

    def test_auto_fallback_warns_when_phase2_flags_set(self, tmp_path, capsys):
        fake_kit = tmp_path / "kit"
        fake_kit.mkdir()
        with patch(
            "bcblib.tools.lesion_features._disco.disco2_ready",
            return_value=None,
        ), patch(
            "bcblib.tools.lesion_features._disco.find_bcbtoolkit",
            return_value=fake_kit,
        ):
            engine, runner = _select_disco_engine(
                _args(engine="auto", len_min=20.0)
            )
        assert engine == "bcbtoolkit"
        assert runner is not None
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "--len-min" in err

    def test_auto_prefers_disco2_no_warning_when_phase2_flags_set(self, tmp_path, capsys):
        with patch(
            "bcblib.tools.lesion_features._disco.disco2_ready",
            return_value=tmp_path,
        ):
            engine, runner = _select_disco_engine(
                _args(engine="auto", fiber_class="association")
            )
        assert engine == "disco2"
        assert runner.keywords["fiber_class"] == "association"
        err = capsys.readouterr().err
        assert "WARNING" not in err


class TestMainMoveIntoBidsStructure:
    """Regression (2026-09-17, BBS_M00 2mm regen on deeper2): main()'s
    move-into-BIDS-structure step predicted disco2's output filename without
    the out_voxel_size that makes disco2 rewrite the res- entity, matched 0
    files, and its `finally` then deleted the flat staging dir -- silently
    discarding all 337 freshly-computed disconnectomes. This exercises main()
    end to end (preprocess_batch and the engine runner mocked; everything
    else, including predict_disco_output and the real move/cleanup logic, is
    the genuine code path)."""

    def _make_output_dir(self, tmp_path, res="1"):
        output_dir = tmp_path / "prep"
        sub_dir = output_dir / "sub-001"
        sub_dir.mkdir(parents=True)
        lesion = sub_dir / f"sub-001_space-MNI_res-{res}_label-lesion_mask.nii.gz"
        lesion.touch()
        return output_dir, lesion

    def test_out_voxel_size_2_moves_disconnectome_into_place(self, tmp_path):
        from bcblib.scripts.run_lf_preprocess import main

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir, lesion = self._make_output_dir(tmp_path)

        def fake_runner(lesion_dir, disco_flat):
            disco_flat.mkdir(parents=True, exist_ok=True)
            # what disco2 actually writes with --out-voxel-size 2: res- rewritten.
            (disco_flat / "sub-001_space-MNI_res-2_desc-disconnectome.nii.gz").touch()
            return {}

        with patch(
            "bcblib.tools.lesion_features._pipeline.preprocess_batch",
            return_value={"001": lesion},
        ), patch(
            "bcblib.scripts.run_lf_preprocess._select_disco_engine",
            return_value=("disco2", fake_runner),
        ):
            main([
                "--bids-dir", str(bids_dir), "--output-dir", str(output_dir),
                "--engine", "disco2", "--out-voxel-size", "2",
            ])

        moved = output_dir / "sub-001" / "sub-001_space-MNI_res-2_desc-disconnectome.nii.gz"
        assert moved.exists()
        assert not (output_dir / "_tmp_disco_flat").exists()
        assert not (output_dir / "_tmp_lesions_for_disco").exists()

    def test_out_voxel_size_2_unmatched_output_is_not_deleted(self, tmp_path, capsys):
        """If a future change makes predict_disco_output diverge from disco2's
        real output again, the computed file must survive (left in
        _tmp_disco_flat) instead of being silently deleted -- the safety net
        added alongside this fix."""
        from bcblib.scripts.run_lf_preprocess import main

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir, lesion = self._make_output_dir(tmp_path)

        def fake_runner(lesion_dir, disco_flat):
            disco_flat.mkdir(parents=True, exist_ok=True)
            # simulate a naming mismatch: disco2 wrote something predict_disco_output
            # cannot map back to sub-001 (e.g. a stale res- entity again).
            (disco_flat / "sub-001_space-MNI_res-9_desc-disconnectome.nii.gz").touch()
            return {}

        with patch(
            "bcblib.tools.lesion_features._pipeline.preprocess_batch",
            return_value={"001": lesion},
        ), patch(
            "bcblib.scripts.run_lf_preprocess._select_disco_engine",
            return_value=("disco2", fake_runner),
        ):
            main([
                "--bids-dir", str(bids_dir), "--output-dir", str(output_dir),
                "--engine", "disco2", "--out-voxel-size", "2",
            ])

        survivor = output_dir / "_tmp_disco_flat" / "sub-001_space-MNI_res-9_desc-disconnectome.nii.gz"
        assert survivor.exists()
        assert "WARNING" in capsys.readouterr().err

    def test_bcbtoolkit_engine_ignores_out_voxel_size_in_move_step(self, tmp_path):
        """out_voxel_size only applies to disco2; under bcbtoolkit it must NOT be
        threaded into predict_disco_output (BCBToolKit never rewrites res-), even
        if the flag was set on the CLI (it's ignored with a warning for the run
        itself -- the move step must agree)."""
        from bcblib.scripts.run_lf_preprocess import main

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir, lesion = self._make_output_dir(tmp_path)

        def fake_runner(lesion_dir, disco_flat):
            disco_flat.mkdir(parents=True, exist_ok=True)
            (disco_flat / "sub-001_space-MNI_res-1_desc-disconnectome.nii.gz").touch()
            return {}

        with patch(
            "bcblib.tools.lesion_features._pipeline.preprocess_batch",
            return_value={"001": lesion},
        ), patch(
            "bcblib.scripts.run_lf_preprocess._select_disco_engine",
            return_value=("bcbtoolkit", fake_runner),
        ):
            main([
                "--bids-dir", str(bids_dir), "--output-dir", str(output_dir),
                "--engine", "bcbtoolkit", "--out-voxel-size", "2",
            ])

        moved = output_dir / "sub-001" / "sub-001_space-MNI_res-1_desc-disconnectome.nii.gz"
        assert moved.exists()

    def test_runner_exception_mid_batch_does_not_delete_computed_output(self, tmp_path):
        """Regression: disco2 batch itself catches per-lesion errors and keeps
        writing every other subject's output, only raising (via run_disco2_batch's
        RuntimeError) after finishing -- so a single failed lesion among many must
        not cost the rest. Before this fix, moved_cleanly didn't exist and the
        `finally` block deleted disco_flat unconditionally whenever `unmatched`
        was empty, which it always was if the exception fired before the
        move/unmatched loop ran at all."""
        from bcblib.scripts.run_lf_preprocess import main

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir, lesion = self._make_output_dir(tmp_path)

        def crashing_runner(lesion_dir, disco_flat):
            disco_flat.mkdir(parents=True, exist_ok=True)
            # one subject's output was already computed and written...
            (disco_flat / "sub-001_space-MNI_res-2_desc-disconnectome.nii.gz").touch()
            # ...before disco2 batch hit a fatal error on some other lesion.
            raise RuntimeError("disco2 batch failed with exit code 1.")

        with patch(
            "bcblib.tools.lesion_features._pipeline.preprocess_batch",
            return_value={"001": lesion},
        ), patch(
            "bcblib.scripts.run_lf_preprocess._select_disco_engine",
            return_value=("disco2", crashing_runner),
        ):
            with pytest.raises(RuntimeError):
                main([
                    "--bids-dir", str(bids_dir), "--output-dir", str(output_dir),
                    "--engine", "disco2", "--out-voxel-size", "2",
                ])

        survivor = output_dir / "_tmp_disco_flat" / "sub-001_space-MNI_res-2_desc-disconnectome.nii.gz"
        assert survivor.exists()
