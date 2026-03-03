"""Tests for the RESET_BEST sentinel-based best-loss reset mechanism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from df_mlx.training_checkpoints import _RESET_BEST_SENTINEL, check_reset_best_sentinel


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """Return a temporary checkpoint directory."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


def _write_fake_best(ckpt_dir: Path) -> tuple[Path, Path]:
    """Create dummy best.safetensors and its .state.json sidecar."""
    weights = ckpt_dir / "best.safetensors"
    state = ckpt_dir / "best.safetensors.state.json"
    weights.write_bytes(b"\x00" * 16)
    state.write_text(json.dumps({"epoch": 5, "best_valid_loss": 0.42}))
    return weights, state


class TestCheckResetBestSentinel:
    def test_no_sentinel_returns_false(self, ckpt_dir: Path) -> None:
        assert check_reset_best_sentinel(ckpt_dir, stage_index=0, epoch=10) is False

    def test_sentinel_triggers_reset(self, ckpt_dir: Path) -> None:
        _write_fake_best(ckpt_dir)
        sentinel = ckpt_dir / _RESET_BEST_SENTINEL
        sentinel.touch()

        result = check_reset_best_sentinel(ckpt_dir, stage_index=1, epoch=9)

        assert result is True
        assert not sentinel.exists(), "sentinel should be consumed"

    def test_best_files_backed_up(self, ckpt_dir: Path) -> None:
        weights, state = _write_fake_best(ckpt_dir)
        (ckpt_dir / _RESET_BEST_SENTINEL).touch()

        check_reset_best_sentinel(ckpt_dir, stage_index=2, epoch=14)

        assert not weights.exists(), "original best.safetensors should be renamed"
        assert not state.exists(), "original state.json should be renamed"

        backup_weights = ckpt_dir / "best_stage2_epoch015.safetensors"
        backup_state = ckpt_dir / "best_stage2_epoch015.safetensors.state.json"
        assert backup_weights.exists()
        assert backup_state.exists()
        assert backup_weights.read_bytes() == b"\x00" * 16
        assert json.loads(backup_state.read_text())["best_valid_loss"] == 0.42

    def test_sentinel_without_best_files(self, ckpt_dir: Path) -> None:
        """Sentinel should still return True even if there's nothing to back up."""
        (ckpt_dir / _RESET_BEST_SENTINEL).touch()

        result = check_reset_best_sentinel(ckpt_dir, stage_index=0, epoch=0)

        assert result is True
        assert not (ckpt_dir / _RESET_BEST_SENTINEL).exists()

    def test_sentinel_consumed_only_once(self, ckpt_dir: Path) -> None:
        (ckpt_dir / _RESET_BEST_SENTINEL).touch()

        assert check_reset_best_sentinel(ckpt_dir, stage_index=0, epoch=0) is True
        assert check_reset_best_sentinel(ckpt_dir, stage_index=0, epoch=1) is False

    def test_backup_naming_includes_stage_and_epoch(self, ckpt_dir: Path) -> None:
        _write_fake_best(ckpt_dir)
        (ckpt_dir / _RESET_BEST_SENTINEL).touch()

        check_reset_best_sentinel(ckpt_dir, stage_index=3, epoch=99)

        assert (ckpt_dir / "best_stage3_epoch100.safetensors").exists()
        assert (ckpt_dir / "best_stage3_epoch100.safetensors.state.json").exists()
