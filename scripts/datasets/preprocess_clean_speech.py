#!/usr/bin/env python3
"""Enhance a clean-speech corpus with DeepFilterNet3 and mirror it to a new tree.

This is intended as an optional pre-step before building the MLX datastore.
It preserves relative paths under ``--base-dir`` and emits a file list that can
be fed directly into ``build_mlx_datastore.sh`` / ``df_mlx.build_audio_cache``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from df.enhance import AudioDataset, enhance, init_df
from df.io import resample, save_audio
from df.model import ModelParams


class PreprocessProgressStats:
    def __init__(self, start_time: float) -> None:
        self.start_time = start_time
        self.enhance_count = 0
        self.enhance_seconds = 0.0
        self.save_count = 0
        self.save_seconds = 0.0
        self.queue_high_water = 0


def read_file_list(path: Path) -> List[Path]:
    files: List[Path] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            files.append(Path(line).expanduser().resolve())
    return files


def build_output_path(source: Path, output_root: Path, base_dir: Path) -> Path:
    try:
        relative = source.relative_to(base_dir)
    except ValueError:
        relative = Path("_external") / source.name
    return output_root / relative


def write_output_list(paths: Iterable[Path], output_list: Path) -> None:
    output_list.parent.mkdir(parents=True, exist_ok=True)
    temp_output_list = output_list.with_name(f"{output_list.name}.tmp.{os.getpid()}")
    with temp_output_list.open("w") as handle:
        for path in paths:
            handle.write(f"{path}\n")
    temp_output_list.replace(output_list)


def is_complete_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def build_temp_output_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.partial.{os.getpid()}")


def resolve_effective_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_save_workers(loader_workers: int, effective_device: str) -> int:
    if effective_device.startswith("cpu"):
        return 1
    return max(2, min(8, max(1, loader_workers)))


def _format_average_ms(total_seconds: float, count: int) -> str:
    if count <= 0:
        return "-"
    return f"{(total_seconds / count) * 1000.0:.0f}ms"


def build_progress_postfix(
    stats: PreprocessProgressStats, inflight_saves: int, *, now: float | None = None
) -> dict[str, str]:
    current_time = time.perf_counter() if now is None else now
    elapsed = max(current_time - stats.start_time, 1e-9)
    return {
        "fps": f"{stats.enhance_count / elapsed:.2f}",
        "save_q": str(inflight_saves),
        "enh": _format_average_ms(stats.enhance_seconds, stats.enhance_count),
        "save": _format_average_ms(stats.save_seconds, stats.save_count),
    }


def save_enhanced_audio_atomically(target: Path, enhanced_audio: torch.Tensor, df_sr: int, orig_sr: int) -> float:
    temp_target = build_temp_output_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if temp_target.exists():
        temp_target.unlink()
    save_started = time.perf_counter()
    try:
        audio_to_save = enhanced_audio
        if orig_sr != df_sr:
            audio_to_save = resample(audio_to_save, df_sr, orig_sr)
        save_audio(str(temp_target), audio_to_save, sr=orig_sr, output_dir=None, suffix=None, log=False)
        temp_target.replace(target)
        return time.perf_counter() - save_started
    except Exception:
        if temp_target.exists():
            temp_target.unlink()
        raise


def collect_completed_saves(
    inflight_saves: dict[Future[float], Path],
    failures: list[str],
    stats: PreprocessProgressStats,
    *,
    block_until_all: bool = False,
) -> None:
    if not inflight_saves:
        return
    return_when = ALL_COMPLETED if block_until_all else FIRST_COMPLETED
    completed, _ = wait(set(inflight_saves), return_when=return_when)
    for future in completed:
        source = inflight_saves.pop(future)
        try:
            stats.save_seconds += future.result()
            stats.save_count += 1
        except Exception as exc:  # pragma: no cover - operational safeguard
            failures.append(f"{source}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess clean speech with DeepFilterNet3.")
    parser.add_argument("--file-list", required=True, help="Input clean-speech file list.")
    parser.add_argument("--output-root", required=True, help="Root directory for enhanced copies.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory used to preserve relative paths under the output root.",
    )
    parser.add_argument(
        "--output-list",
        required=True,
        help="Path to write the enhanced file list for downstream datastore building.",
    )
    parser.add_argument(
        "--model-base-dir",
        default="DeepFilterNet3",
        help="Pretrained model name or model directory (default: DeepFilterNet3).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device override: cpu, cuda, mps, etc.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers used while reading source audio (resume is default unless --overwrite is set).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even when the mirrored file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    file_list = Path(args.file_list).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_list = Path(args.output_list).expanduser().resolve()
    effective_device = resolve_effective_device(args.device)

    sources = read_file_list(file_list)
    if not sources:
        raise SystemExit(f"No input files found in {file_list}")

    output_root.mkdir(parents=True, exist_ok=True)

    output_paths = [build_output_path(source, output_root, base_dir) for source in sources]
    pending_sources = [
        source for source, target in zip(sources, output_paths) if args.overwrite or not is_complete_output(target)
    ]
    save_workers = choose_save_workers(args.num_workers, effective_device)

    print("=" * 60)
    print("Clean Speech Preprocessor")
    print("=" * 60)
    print(f"Input list:      {file_list}")
    print(f"Input files:     {len(sources):,}")
    print(f"Pending files:   {len(pending_sources):,}")
    print(f"Output root:     {output_root}")
    print(f"Output list:     {output_list}")
    print(f"Base dir:        {base_dir}")
    print(f"Model:           {args.model_base_dir}")
    print(f"Device:          {effective_device}")
    print(f"Workers:         {args.num_workers}")
    print(f"Save workers:    {save_workers}")
    print(f"Mode:            {'overwrite' if args.overwrite else 'resume'}")
    print("=" * 60)

    if pending_sources:
        model, df_state, _, _ = init_df(
            model_base_dir=args.model_base_dir,
            post_filter=False,
            log_level="INFO",
            log_file=None,
            config_allow_defaults=True,
            epoch="best",
            default_model="DeepFilterNet3",
            mask_only=False,
            device=args.device,
        )

        df_sr = ModelParams().sr
        dataset = AudioDataset([str(path) for path in pending_sources], df_sr)
        loader_kwargs: dict[str, object] = {
            "num_workers": max(0, args.num_workers),
            "pin_memory": effective_device.startswith("cuda"),
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = min(8, max(2, args.num_workers))
        loader = DataLoader(dataset, **loader_kwargs)

        failures: list[str] = []
        inflight_saves: dict[Future[float], Path] = {}
        max_inflight_saves = max(2, save_workers * 2)
        progress_stats = PreprocessProgressStats(start_time=time.perf_counter())
        with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
            with torch.inference_mode():
                with tqdm(
                    loader,
                    total=len(dataset),
                    desc="Enhancing",
                    unit="file",
                    dynamic_ncols=True,
                ) as progress:
                    for file_batch, audio_batch, orig_sr_batch in progress:
                        source = Path(file_batch[0]).expanduser().resolve()
                        target = build_output_path(source, output_root, base_dir)
                        try:
                            audio = audio_batch.squeeze(0)
                            orig_sr = int(orig_sr_batch[0])
                            enhance_started = time.perf_counter()
                            enhanced = enhance(model, df_state, audio, pad=True, device=effective_device).detach().cpu()
                            progress_stats.enhance_seconds += time.perf_counter() - enhance_started
                            progress_stats.enhance_count += 1
                            future = save_pool.submit(save_enhanced_audio_atomically, target, enhanced, df_sr, orig_sr)
                            inflight_saves[future] = source
                            progress_stats.queue_high_water = max(progress_stats.queue_high_water, len(inflight_saves))
                            if len(inflight_saves) >= max_inflight_saves:
                                collect_completed_saves(inflight_saves, failures, progress_stats)
                            progress.set_postfix(build_progress_postfix(progress_stats, len(inflight_saves)))
                        except Exception as exc:  # pragma: no cover - operational safeguard
                            failures.append(f"{source}: {exc}")
                            progress.set_postfix(build_progress_postfix(progress_stats, len(inflight_saves)))
                    collect_completed_saves(inflight_saves, failures, progress_stats, block_until_all=True)
                    progress.set_postfix(build_progress_postfix(progress_stats, len(inflight_saves)))

        elapsed = max(time.perf_counter() - progress_stats.start_time, 1e-9)
        print(
            "Preprocess summary: "
            f"{progress_stats.enhance_count:,} enhanced in {elapsed:.1f}s "
            f"({progress_stats.enhance_count / elapsed:.2f} files/s) | "
            f"avg enhance {_format_average_ms(progress_stats.enhance_seconds, progress_stats.enhance_count)} | "
            f"avg save {_format_average_ms(progress_stats.save_seconds, progress_stats.save_count)} | "
            f"save queue high-water {progress_stats.queue_high_water}"
        )

        if failures:
            print("The following files failed to preprocess:", file=sys.stderr)
            for item in failures[:20]:
                print(f"  {item}", file=sys.stderr)
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
            raise SystemExit(1)
    else:
        print("All mirrored outputs already exist; reusing them.")

    write_output_list(output_paths, output_list)
    print(f"Wrote output list with {len(output_paths):,} entries -> {output_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
