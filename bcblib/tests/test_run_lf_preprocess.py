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
