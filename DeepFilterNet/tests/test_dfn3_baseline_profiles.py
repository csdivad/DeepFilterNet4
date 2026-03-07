from __future__ import annotations

import sys
from pathlib import Path

# Ensure the df_mlx package is importable when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from df_mlx.run_config import load_run_config  # noqa: E402

PROFILE_DIR = Path(__file__).resolve().parents[1] / "df_mlx" / "configs" / "run_profiles"


def test_full_vadlite_profile_exists_and_loads() -> None:
    path = PROFILE_DIR / "baseline_dfn3_gan_vad_speech_full_vadlite.toml"
    assert path.exists(), f"Missing run profile: {path}"

    cfg = load_run_config(path)

    assert cfg.loss.dynamic_loss == "baseline"
    assert cfg.loss.pipeline_stages == []
    assert cfg.loss.mrstft.factor == 0.0
    assert cfg.vad.loss_weight > 0.0
    assert cfg.vad.speech_loss_weight > 0.0
    assert cfg.gan.enabled is True
    assert cfg.gan.start_epoch > 0
    assert cfg.gan.ramp_epochs > 0
    assert "full_vadlite" in cfg.checkpoint.checkpoint_dir


def test_full_vadlite_profile_matches_vadlite_ablation_core_settings() -> None:
    ablation_cfg = load_run_config(PROFILE_DIR / "baseline_dfn3_gan_vad_speech_ablation_vadlite.toml")
    full_cfg = load_run_config(PROFILE_DIR / "baseline_dfn3_gan_vad_speech_full_vadlite.toml")

    assert full_cfg.loss.dynamic_loss == ablation_cfg.loss.dynamic_loss == "baseline"
    assert full_cfg.vad.loss_weight == ablation_cfg.vad.loss_weight == 0.20
    assert full_cfg.vad.speech_loss_weight == ablation_cfg.vad.speech_loss_weight == 0.20
    assert full_cfg.gan.start_epoch == ablation_cfg.gan.start_epoch == 50
    assert full_cfg.gan.ramp_epochs == ablation_cfg.gan.ramp_epochs == 24
    assert full_cfg.gan.adv_weight == ablation_cfg.gan.adv_weight == 0.08
    assert full_cfg.gan.fm_weight == ablation_cfg.gan.fm_weight == 0.6
    assert full_cfg.training.epochs > ablation_cfg.training.epochs
    assert full_cfg.checkpoint.save_total_limit is not None
    assert ablation_cfg.checkpoint.save_total_limit is not None
    assert full_cfg.checkpoint.save_total_limit > ablation_cfg.checkpoint.save_total_limit
