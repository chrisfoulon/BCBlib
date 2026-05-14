"""Tests for bcblib.scripts.imaging_cli entry points."""

import json
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


class TestBcbInfoMulti:
    """Multi-image features of bcb-info."""

    def test_short_multiple_images(self, tmp_path, capsys):
        p1 = _make_nii(tmp_path, name="a.nii.gz", shape=(4, 5, 6))
        p2 = _make_nii(tmp_path, name="b.nii.gz", shape=(2, 3, 4))
        from bcblib.scripts.imaging_cli import bcb_info
        sys.argv = ["bcb-info", "--short", str(p1), str(p2)]
        bcb_info()
        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l.strip()]
        assert len(lines) == 2
        assert "a.nii.gz" in lines[0]
        assert "b.nii.gz" in lines[1]

    def test_multi_produces_output_for_each(self, tmp_path, capsys):
        p1 = _make_nii(tmp_path, name="x.nii.gz", shape=(4, 5, 6))
        p2 = _make_nii(tmp_path, name="y.nii.gz", shape=(2, 3, 4))
        from bcblib.scripts.imaging_cli import bcb_info
        sys.argv = ["bcb-info", "--no-color", str(p1), str(p2)]
        bcb_info()
        out = capsys.readouterr().out
        # Both filenames should appear somewhere in the output
        assert "x.nii.gz" in out
        assert "y.nii.gz" in out


class TestBcbHeader:
    def test_single_image(self, tmp_path, capsys):
        p = _make_nii(tmp_path)
        from bcblib.scripts.imaging_cli import bcb_header
        sys.argv = ["bcb-header", str(p)]
        bcb_header()
        out = capsys.readouterr().out
        assert "dim" in out

    def test_single_image_json(self, tmp_path, capsys):
        p = _make_nii(tmp_path)
        from bcblib.scripts.imaging_cli import bcb_header
        sys.argv = ["bcb-header", "--json", str(p)]
        bcb_header()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "dim" in data

    def test_multi_comparison_contains_both_names(self, tmp_path, capsys):
        p1 = _make_nii(tmp_path, name="a.nii.gz", shape=(4, 5, 6))
        p2 = _make_nii(tmp_path, name="b.nii.gz", shape=(2, 3, 4))
        from bcblib.scripts.imaging_cli import bcb_header
        sys.argv = ["bcb-header", str(p1), str(p2)]
        bcb_header()
        out = capsys.readouterr().out
        assert "a.nii.gz" in out
        assert "b.nii.gz" in out

    def test_multi_sequential_flag(self, tmp_path, capsys):
        p1 = _make_nii(tmp_path, name="a.nii.gz", shape=(4, 5, 6))
        p2 = _make_nii(tmp_path, name="b.nii.gz", shape=(2, 3, 4))
        from bcblib.scripts.imaging_cli import bcb_header
        sys.argv = ["bcb-header", "--sequential", str(p1), str(p2)]
        bcb_header()
        out = capsys.readouterr().out
        assert "a.nii.gz" in out
        assert "b.nii.gz" in out

    def test_multi_json_is_dict_keyed_by_filename(self, tmp_path, capsys):
        p1 = _make_nii(tmp_path, name="a.nii.gz")
        p2 = _make_nii(tmp_path, name="b.nii.gz")
        from bcblib.scripts.imaging_cli import bcb_header
        sys.argv = ["bcb-header", "--json", str(p1), str(p2)]
        bcb_header()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert str(p1) in data
        assert str(p2) in data


class TestBcbOrient:
    def test_query(self, tmp_path):
        p = _make_nii(tmp_path)
        from bcblib.scripts.imaging_cli import bcb_orient
        sys.argv = ["bcb-orient", str(p)]
        bcb_orient()  # prints orientation, should not raise

    def test_set_orientation(self, tmp_path, capsys):
        """--set should reorient and report the new orientation."""
        p = _make_nii(tmp_path, shape=(6, 7, 8))
        from bcblib.scripts.imaging_cli import bcb_orient
        sys.argv = ["bcb-orient", "--set", "RAS", str(p)]
        bcb_orient()
        out = capsys.readouterr().out
        # Output should confirm the save path
        assert str(p) in out

    def test_set_with_output(self, tmp_path, capsys):
        """-o should write to a separate output file."""
        src = _make_nii(tmp_path, name="src.nii.gz", shape=(6, 7, 8))
        out_path = tmp_path / "reoriented.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_orient
        sys.argv = ["bcb-orient", "--set", "RAS",
                    "-o", str(out_path), str(src)]
        bcb_orient()
        out = capsys.readouterr().out
        assert str(out_path) in out
        assert out_path.exists()
        # Source file should be unchanged
        import nibabel as nib
        orig = nib.load(str(src))
        assert orig.shape == (6, 7, 8)


class TestBcbStatsMask:
    def test_mask(self, tmp_path, capsys):
        """--mask should restrict stats to masked voxels."""
        data = np.zeros((6, 6, 6), dtype=np.float32)
        data[1:4, 1:4, 1:4] = 10.0
        img = tmp_path / "data.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(img))

        mask_data = np.zeros((6, 6, 6), dtype=np.uint8)
        mask_data[1:4, 1:4, 1:4] = 1
        mask = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), str(mask))

        from bcblib.scripts.imaging_cli import bcb_stats
        sys.argv = ["bcb-stats", "--json", "--mask", str(mask), str(img)]
        bcb_stats()
        out = capsys.readouterr().out
        stats = json.loads(out)
        # All non-zero voxels in masked region should be 10
        assert abs(stats["mean"] - 10.0) < 1e-4
        assert stats["min"] == pytest.approx(10.0, abs=1e-4)


class TestBcbRoi:
    def test_basic_spatial_crop(self, tmp_path, capsys):
        """bcb-roi should produce a sub-volume with the specified extents."""
        src = _make_nii(tmp_path, name="src.nii.gz", shape=(10, 10, 10))
        out = tmp_path / "roi.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_roi
        sys.argv = [
            "bcb-roi",
            str(src), str(out),
            "1", "4",   # x: 1..4
            "2", "3",   # y: 2..4
            "0", "5",   # z: 0..4
        ]
        bcb_roi()
        assert out.exists()
        import nibabel as nib
        result = nib.load(str(out))
        assert result.shape == (4, 3, 5)

    def test_print_message(self, tmp_path, capsys):
        """bcb-roi should print a confirmation message."""
        src = _make_nii(tmp_path, name="src.nii.gz", shape=(8, 8, 8))
        out = tmp_path / "roi.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_roi
        sys.argv = [
            "bcb-roi",
            str(src), str(out),
            "0", "4", "0", "4", "0", "4",
        ]
        bcb_roi()
        stdout = capsys.readouterr().out
        assert "Saved ROI" in stdout
        assert str(out) in stdout


class TestBcbMerge:
    def test_merge_two_3d(self, tmp_path, capsys):
        """bcb-merge should concatenate two 3-D images to a 4-D volume."""
        p1 = _make_nii(tmp_path, name="vol0.nii.gz", shape=(4, 5, 6), val=1.0)
        p2 = _make_nii(tmp_path, name="vol1.nii.gz", shape=(4, 5, 6), val=2.0)
        out = tmp_path / "merged.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_merge
        sys.argv = ["bcb-merge", str(out), str(p1), str(p2)]
        bcb_merge()
        assert out.exists()
        import nibabel as nib
        result = nib.load(str(out))
        assert result.shape == (4, 5, 6, 2)

    def test_print_message(self, tmp_path, capsys):
        """bcb-merge should print a confirmation message."""
        p1 = _make_nii(tmp_path, name="a.nii.gz", shape=(3, 3, 3))
        p2 = _make_nii(tmp_path, name="b.nii.gz", shape=(3, 3, 3))
        out = tmp_path / "m.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_merge
        sys.argv = ["bcb-merge", str(out), str(p1), str(p2)]
        bcb_merge()
        stdout = capsys.readouterr().out
        assert "Merged 2 images" in stdout

    def test_merge_along_axis2(self, tmp_path):
        """--axis flag should merge along a non-default axis."""
        p1 = _make_nii(tmp_path, name="c.nii.gz", shape=(4, 5, 6))
        p2 = _make_nii(tmp_path, name="d.nii.gz", shape=(4, 5, 6))
        out = tmp_path / "ax2.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_merge
        sys.argv = ["bcb-merge", "--axis", "2", str(out), str(p1), str(p2)]
        bcb_merge()
        import nibabel as nib
        result = nib.load(str(out))
        assert result.shape == (4, 5, 12)


class TestBcbSplit:
    def _make_4d(self, tmp_path, name="4d.nii.gz", volumes=3, shape=(4, 5, 6)):
        data = np.stack(
            [np.full(shape, float(i), dtype=np.float32) for i in range(volumes)],
            axis=-1,
        )
        p = tmp_path / name
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
        return p

    def test_split_produces_correct_count(self, tmp_path, capsys):
        """bcb-split should write one file per volume."""
        src = self._make_4d(tmp_path, volumes=3)
        prefix = str(tmp_path / "vol_")
        from bcblib.scripts.imaging_cli import bcb_split
        sys.argv = ["bcb-split", str(src), prefix]
        bcb_split()
        files = sorted(tmp_path.glob("vol_*.nii.gz"))
        assert len(files) == 3

    def test_split_volumes_are_3d(self, tmp_path):
        """Each output volume should be 3-D."""
        src = self._make_4d(tmp_path, volumes=2, shape=(4, 5, 6))
        prefix = str(tmp_path / "v_")
        from bcblib.scripts.imaging_cli import bcb_split
        sys.argv = ["bcb-split", str(src), prefix]
        bcb_split()
        for f in sorted(tmp_path.glob("v_*.nii.gz")):
            vol = nib.load(str(f))
            assert vol.ndim == 3
            assert vol.shape == (4, 5, 6)

    def test_print_message(self, tmp_path, capsys):
        """bcb-split should print a confirmation message."""
        src = self._make_4d(tmp_path, volumes=2)
        prefix = str(tmp_path / "s_")
        from bcblib.scripts.imaging_cli import bcb_split
        sys.argv = ["bcb-split", str(src), prefix]
        bcb_split()
        stdout = capsys.readouterr().out
        assert "Split into 2 volumes" in stdout


class TestBcbConvert:
    def test_nii_to_nii_gz(self, tmp_path, capsys):
        """bcb-convert should compress .nii to .nii.gz."""
        data = np.ones((3, 4, 5), dtype=np.float32)
        src = tmp_path / "img.nii"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(src))
        out = tmp_path / "img.nii.gz"
        from bcblib.scripts.imaging_cli import bcb_convert
        sys.argv = ["bcb-convert", str(src), str(out)]
        bcb_convert()
        assert out.exists()
        result = nib.load(str(out))
        assert result.shape == (3, 4, 5)

    def test_nii_gz_to_nii(self, tmp_path):
        """bcb-convert should decompress .nii.gz to .nii."""
        src = _make_nii(tmp_path, name="img.nii.gz", shape=(3, 4, 5))
        out = tmp_path / "img.nii"
        from bcblib.scripts.imaging_cli import bcb_convert
        sys.argv = ["bcb-convert", str(src), str(out)]
        bcb_convert()
        assert out.exists()
        result = nib.load(str(out))
        assert result.shape == (3, 4, 5)

    def test_print_message(self, tmp_path, capsys):
        """bcb-convert should print a confirmation message."""
        src = _make_nii(tmp_path, name="orig.nii.gz", shape=(2, 2, 2))
        out = tmp_path / "orig.nii"
        from bcblib.scripts.imaging_cli import bcb_convert
        sys.argv = ["bcb-convert", str(src), str(out)]
        bcb_convert()
        stdout = capsys.readouterr().out
        assert "Converted" in stdout
        assert str(out) in stdout
