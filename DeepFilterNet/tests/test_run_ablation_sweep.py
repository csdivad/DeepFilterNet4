from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure the df_mlx package is importable when running tests from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from df_mlx.run_ablation_sweep import DEFAULT_ABLATION_PROFILES, _resolve_profiles  # noqa: E402

PROFILE_DIR = Path(__file__).resolve().parents[1] / "df_mlx" / "configs" / "run_profiles"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "df_mlx" / "run_ablation_sweep.py"


def test_default_ablation_profiles_exist() -> None:
    for name in DEFAULT_ABLATION_PROFILES:
        path = PROFILE_DIR / name
        assert path.exists(), f"Missing default ablation profile: {path}"


def test_resolve_profiles_deduplicates_by_variant_name() -> None:
    variants = _resolve_profiles(
        [
            "baseline_dfn3_gan_vad_speech_ablation_vadlite.toml",
            "baseline_dfn3_gan_vad_speech_ablation_vadlite.toml",
            "baseline_dfn3_gan_vad_speech_ablation_ganmix.toml",
        ],
        PROFILE_DIR,
    )

    assert [variant.name for variant in variants] == [
        "baseline_dfn3_gan_vad_speech_ablation_vadlite",
        "baseline_dfn3_gan_vad_speech_ablation_ganmix",
    ]


def test_run_ablation_sweep_dry_run_executes(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--checkpoint-root",
            str(tmp_path / "ablation_ckpts"),
            "--profiles",
            "baseline_dfn3_gan_vad_speech_ablation_vadlite.toml",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Resumable DFN3+GAN ablation sweep" in result.stdout
    assert "Dry-run only; command not executed." in result.stdout
    assert "baseline_dfn3_gan_vad_speech_ablation_vadlite" in result.stdout
