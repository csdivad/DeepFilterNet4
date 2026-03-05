#!/usr/bin/env python3
"""Run resumable DFN3+GAN ablation sweeps with capped batches.

This helper is designed for fast, comparable screening runs across multiple
run profiles without losing progress on interruption.

Key behavior:
- Per-variant checkpoint directories under ``--checkpoint-root``.
- Auto-resume from each variant directory when rerun.
- Optional one-time seed resume from a shared checkpoint when variant
  directory is empty.
- Epoch-level caps via ``--max-train-batches`` and ``--max-valid-batches``
  to reduce wall time.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_ABLATION_PROFILES = (
    "baseline_dfn3_gan_vad_speech_ablation_vadlite.toml",
    "baseline_dfn3_gan_vad_speech_ablation_ganmix.toml",
    "baseline_dfn3_gan_vad_speech_ablation_datahard.toml",
)


@dataclass(frozen=True)
class Variant:
    name: str
    profile_path: Path


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the newest valid checkpoint in ``checkpoint_dir`` if any."""
    if not checkpoint_dir.exists():
        return None

    candidates: list[Path] = []
    for ckpt in checkpoint_dir.glob("*.safetensors"):
        name = ckpt.name
        if name.endswith(".disc.safetensors"):
            continue
        if name.startswith("tmp_"):
            continue
        state_path = ckpt.with_suffix(".state.json")
        if not ckpt.exists() or ckpt.stat().st_size == 0:
            continue
        if not state_path.exists() or state_path.stat().st_size == 0:
            continue
        candidates.append(ckpt)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _expand(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def _resolve_profiles(raw_profiles: Iterable[str], profile_dir: Path) -> list[Variant]:
    variants: list[Variant] = []
    for entry in raw_profiles:
        raw = entry.strip()
        if not raw:
            continue
        if "/" in raw or raw.endswith(".toml") and Path(raw).exists():
            profile_path = _expand(raw)
        else:
            profile_path = (profile_dir / raw).resolve()
        if not profile_path.exists():
            raise FileNotFoundError(f"Run profile not found: {profile_path}")
        if profile_path.suffix != ".toml":
            raise ValueError(f"Run profile must be a TOML file: {profile_path}")
        variants.append(Variant(name=profile_path.stem, profile_path=profile_path))

    if not variants:
        raise ValueError("No profiles resolved. Provide at least one profile.")

    # Preserve first occurrence order and deduplicate by variant name.
    deduped: dict[str, Variant] = {}
    for variant in variants:
        deduped.setdefault(variant.name, variant)
    return list(deduped.values())


def _build_train_command(
    *,
    python_bin: str,
    variant: Variant,
    ckpt_dir: Path,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_valid_batches: int,
    seed_checkpoint: Path | None,
    seed_data_checkpoint: Path | None,
    gan_start_epoch: int | None,
    gan_ramp_epochs: int | None,
) -> list[str]:
    cmd: list[str] = [
        python_bin,
        "-m",
        "df_mlx.train_dynamic",
        "--run-config",
        str(variant.profile_path),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--max-train-batches",
        str(max_train_batches),
        "--max-valid-batches",
        str(max_valid_batches),
    ]

    if gan_start_epoch is not None:
        cmd.extend(["--gan-start-epoch", str(gan_start_epoch)])
    if gan_ramp_epochs is not None:
        cmd.extend(["--gan-ramp-epochs", str(gan_ramp_epochs)])

    # Resume policy:
    # 1) Prefer local variant checkpoint/data if any exist.
    # 2) If variant dir is empty and a shared seed is provided, use it once.
    # 3) Otherwise request auto-resume (will no-op when no ckpt exists).
    latest_local = _find_latest_checkpoint(ckpt_dir)
    has_local_data = (ckpt_dir / "data_checkpoint.json").exists()
    if latest_local is not None:
        cmd.append("--resume")
        if has_local_data:
            cmd.append("--resume-data")
    elif seed_checkpoint is not None:
        cmd.extend(["--resume", str(seed_checkpoint)])
        if seed_data_checkpoint is not None:
            cmd.extend(["--resume-data", str(seed_data_checkpoint)])
    else:
        cmd.extend(["--resume", "--resume-data"])

    return cmd


def _epoch_complete_marker(ckpt_dir: Path, epoch: int) -> Path:
    return ckpt_dir / f"epoch_{epoch:03d}.complete"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _parse_args() -> argparse.Namespace:
    this_dir = Path(__file__).resolve().parent
    default_profile_dir = this_dir / "configs" / "run_profiles"

    parser = argparse.ArgumentParser(
        description="Run resumable short ablation sweeps for DFN3+GAN+VAD profiles.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_ABLATION_PROFILES),
        help=("Run profile file names (resolved in df_mlx/configs/run_profiles) " "or absolute/relative TOML paths."),
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=str(default_profile_dir),
        help="Directory used to resolve profile file names.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="~/DataDump/checkpoints/dfn3_gan_ablation_fast",
        help="Root dir; each variant gets a subdir here.",
    )
    parser.add_argument(
        "--seed-checkpoint",
        type=str,
        default=None,
        help="Optional shared checkpoint used when a variant has no local checkpoint yet.",
    )
    parser.add_argument(
        "--seed-data-checkpoint",
        type=str,
        default=None,
        help="Optional shared data_checkpoint.json paired with --seed-checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=64, help="Total epochs for each run.")
    parser.add_argument("--batch-size", type=int, default=24, help="Mini-batch size.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=192,
        help="Cap training batches per epoch for faster screening.",
    )
    parser.add_argument(
        "--max-valid-batches",
        type=int,
        default=24,
        help="Cap validation batches per validation pass.",
    )
    parser.add_argument(
        "--gan-start-epoch",
        type=int,
        default=None,
        help="Optional override for GAN start epoch (0-based).",
    )
    parser.add_argument(
        "--gan-ramp-epochs",
        type=int,
        default=None,
        help="Optional override for GAN ramp length.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env var for child runs. Repeatable.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip variants that already have epoch_{epochs}.complete in their checkpoint dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit without starting training.",
    )

    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be > 0")
    if args.max_valid_batches <= 0:
        raise ValueError("--max-valid-batches must be > 0")

    return args


def main() -> int:
    args = _parse_args()
    package_root = Path(__file__).resolve().parent.parent

    profile_dir = _expand(args.profile_dir)
    checkpoint_root = _expand(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    seed_checkpoint = _expand(args.seed_checkpoint) if args.seed_checkpoint else None
    seed_data_checkpoint = _expand(args.seed_data_checkpoint) if args.seed_data_checkpoint else None

    if seed_checkpoint is not None and not seed_checkpoint.exists():
        raise FileNotFoundError(f"Seed checkpoint not found: {seed_checkpoint}")
    if seed_data_checkpoint is not None and not seed_data_checkpoint.exists():
        raise FileNotFoundError(f"Seed data checkpoint not found: {seed_data_checkpoint}")

    variants = _resolve_profiles(args.profiles, profile_dir)

    child_env = os.environ.copy()
    child_env.setdefault("DFNET_TQDM_PANELS", "1")
    for kv in args.env:
        if "=" not in kv:
            raise ValueError(f"Invalid --env value (expected KEY=VALUE): {kv}")
        key, value = kv.split("=", 1)
        child_env[key] = value

    print("=" * 76)
    print("Resumable DFN3+GAN ablation sweep")
    print("=" * 76)
    print(f"Profiles:          {', '.join(v.name for v in variants)}")
    print(f"Checkpoint root:   {checkpoint_root}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Max train batches: {args.max_train_batches}")
    print(f"Max valid batches: {args.max_valid_batches}")
    if args.gan_start_epoch is not None:
        print(f"GAN start epoch:   {args.gan_start_epoch}")
    if args.gan_ramp_epochs is not None:
        print(f"GAN ramp epochs:   {args.gan_ramp_epochs}")
    if seed_checkpoint is not None:
        print(f"Seed checkpoint:   {seed_checkpoint}")
    if seed_data_checkpoint is not None:
        print(f"Seed data ckpt:    {seed_data_checkpoint}")
    print()

    completed: list[str] = []
    skipped: list[str] = []

    for idx, variant in enumerate(variants, start=1):
        ckpt_dir = checkpoint_root / variant.name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        marker = _epoch_complete_marker(ckpt_dir, args.epochs)
        if args.skip_complete and marker.exists():
            print(f"[{idx}/{len(variants)}] Skipping {variant.name} (found {marker.name})")
            skipped.append(variant.name)
            continue

        cmd = _build_train_command(
            python_bin=args.python,
            variant=variant,
            ckpt_dir=ckpt_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_batches=args.max_train_batches,
            max_valid_batches=args.max_valid_batches,
            seed_checkpoint=seed_checkpoint,
            seed_data_checkpoint=seed_data_checkpoint,
            gan_start_epoch=args.gan_start_epoch,
            gan_ramp_epochs=args.gan_ramp_epochs,
        )

        print(f"[{idx}/{len(variants)}] Variant: {variant.name}")
        print(f"  Profile:   {variant.profile_path}")
        print(f"  Ckpt dir:  {ckpt_dir}")
        print(f"  Command:   {_format_cmd(cmd)}")

        if args.dry_run:
            print("  Dry-run only; command not executed.")
            print()
            continue

        try:
            result = subprocess.run(cmd, env=child_env, cwd=package_root, check=False)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Re-run this command to resume from checkpoints.")
            return 130

        if result.returncode != 0:
            print(f"\nVariant {variant.name} failed with exit code {result.returncode}.")
            print("Re-run this script to resume from the last saved checkpoint.")
            return result.returncode

        completed.append(variant.name)
        print(f"  Completed: {variant.name}\n")

    print("=" * 76)
    print("Sweep finished")
    print("=" * 76)
    if completed:
        print(f"Completed variants: {', '.join(completed)}")
    if skipped:
        print(f"Skipped variants:   {', '.join(skipped)}")
    if args.dry_run:
        print("No runs executed (--dry-run).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
