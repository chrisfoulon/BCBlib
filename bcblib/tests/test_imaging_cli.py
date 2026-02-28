"""Tests for bcblib.scripts.imaging_cli entry points."""

import os
import subprocess
import sys

import numpy as np
import nibabel as nib
import pytest


def _make_nii(tmp_path, name="test.nii.gz", shape=(4, 5, 6), val=1.0):
    p = tmp_path / name
    data = np.full(shape, val, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    return p


class TestBcbInfo:
    def test_runs(self, tmp_path):
        p = _make_nii(tmp_path)
        from bcblib.scripts.imaging_cli import bcb_info
        sys.argv = ["bcb-info", str(p)]
        bcb_info()  # should not raise

    def test_short_flag(self, tmp_path, capsys):
        p = _make_nii(tmp_path)
        from bcblib.scripts.imaging_cli import bcb_info
        sys.argv = ["bcb-info", "--short", str(p)]
        bcb_info()
        out = capsys.readouterr().out
        # One-liner should contain filename, dtype, and dims on one line
        assert "test.nii.gz" in out
        assert "float32" in out
        assert "[4, 5, 6]" in out

    def test_no_color_flag(self, tmp_path, capsys, monkeypatch):
        p = _make_nii(tmp_path)
        # Clear NO_COLOR in case it's set
        monkeypatch.delenv("NO_COLOR", raising=False)
        from bcblib.scripts.imaging_cli import bcb_info
        sys.argv = ["bcb-info", "--no-color", str(p)]
        bcb_info()
        out = capsys.readouterr().out
        assert "\033[" not in out


class TestBcbStats:
    def test_runs(self, tmp_path):
        p = _make_nii(tmp_path, val=5.0)
        from bcblib.scripts.imaging_cli import bcb_stats
        sys.argv = ["bcb-stats", str(p)]
        bcb_stats()  # should not raise

    def test_grouped_output(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        p = _make_nii(tmp_path, val=5.0)
        from bcblib.scripts.imaging_cli import bcb_stats
        sys.argv = ["bcb-stats", str(p)]
        bcb_stats()
        out = capsys.readouterr().out
        assert "Statistics for" in out
        assert "voxels" in out
        assert "mm\u00b3" in out

    def test_json_unchanged(self, tmp_path, capsys):
        p = _make_nii(tmp_path, val=5.0)
        from bcblib.scripts.imaging_cli import bcb_stats
        sys.argv = ["bcb-stats", "--json", str(p)]
        bcb_stats()
        import json
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "min" in data
        assert "max" in data


class TestBcbOrient:
    def test_query(self, tmp_path):
        p = _make_nii(tmp_path)
        from bcblib.scripts.imaging_cli import bcb_orient
        sys.argv = ["bcb-orient", str(p)]
        bcb_orient()  # prints orientation, should not raise
