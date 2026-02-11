#!/usr/bin/env python3
"""Benchmark MLX dynamic data pipeline throughput and latency tails.

This benchmark targets the data loading path used by df_mlx training:
- PrefetchDataLoader (thread pool prefetch)
- MLXDataStream (mlx-data prefetch) when available

It reports:
- Mean/p50/p95/p99 batch fetch latency
- Batches/s and samples/s throughput
- Total measured batches/samples

Example:
    python -m df_mlx.benchmark_pipeline \
        --cache-dir /path/to/audio_cache \
        --batch-size 8 \
        --batches 200 \
        --workers 1,2,4,8 \
        --backends prefetch,mlx_stream
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mlx.core as mx
import numpy as np

from df_mlx.dynamic_dataset import (
    HAS_MLX_DATA,
    DatasetConfig,
    DynamicDataset,
    MLXDataStream,
    PrefetchDataLoader,
    read_file_list,
)


@dataclass
class BenchmarkResult:
    """Latency and throughput summary for one loader configuration."""

    backend: str
    workers: int
    prefetch: int
    batch_size: int
    repeats: int
    measured_batches: int
    measured_samples: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_seconds: float
    batches_per_sec: float
    samples_per_sec: float


def parse_worker_list(value: str) -> List[int]:
    workers: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        worker = int(token)
        if worker < 1:
            raise ValueError(f"Worker count must be >= 1, got {worker}")
        workers.append(worker)
    if not workers:
        raise ValueError("At least one worker value is required")
    return workers


def parse_backend_list(value: str) -> List[str]:
    valid = {"prefetch", "mlx_stream"}
    backends = [v.strip() for v in value.split(",") if v.strip()]
    if not backends:
        raise ValueError("At least one backend is required")
    invalid = [b for b in backends if b not in valid]
    if invalid:
        raise ValueError(f"Invalid backends: {invalid}. Valid values: {sorted(valid)}")
    return backends


def _safe_percentile(latencies_ms: List[float], q: float) -> float:
    if not latencies_ms:
        return math.nan
    return float(np.percentile(latencies_ms, q))


def _materialize_batch(batch: Dict[str, mx.array]) -> None:
    # Synchronize all arrays to include host->device staging costs in latency.
    mx.eval(*batch.values())


def _batch_size_from_batch(batch: Dict[str, mx.array]) -> int:
    snr = batch.get("snr")
    if snr is not None:
        return int(snr.shape[0])
    first_val = next(iter(batch.values()))
    return int(first_val.shape[0])


def _benchmark_loader_once(
    loader: Iterable[Dict[str, mx.array]],
    warmup_batches: int,
    measured_batches: int,
    sync_arrays: bool,
) -> Dict[str, Any]:
    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if sync_arrays:
            _materialize_batch(batch)

    latencies_ms: List[float] = []
    samples = 0
    batches = 0
    start = time.perf_counter()

    while batches < measured_batches:
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if sync_arrays:
            _materialize_batch(batch)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        samples += _batch_size_from_batch(batch)
        batches += 1

    elapsed = time.perf_counter() - start
    return {
        "latencies_ms": latencies_ms,
        "samples": samples,
        "batches": batches,
        "elapsed_s": elapsed,
    }


def _build_dataset(args: argparse.Namespace) -> DynamicDataset:
    if args.cache_dir is None and (args.speech_list is None or args.noise_list is None):
        raise ValueError("Provide either --cache-dir or both --speech-list and --noise-list")

    speech_files: List[str] = []
    noise_files: List[str] = []
    rir_files: List[str] = []

    if args.speech_list:
        speech_files = read_file_list(args.speech_list)
    if args.noise_list:
        noise_files = read_file_list(args.noise_list)
    if args.rir_list:
        rir_files = read_file_list(args.rir_list)

    config = DatasetConfig(
        cache_dir=args.cache_dir,
        speech_files=speech_files,
        noise_files=noise_files,
        rir_files=rir_files,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        nb_erb=args.nb_erb,
        nb_df=args.nb_df,
        seed=args.seed,
    )
    dataset = DynamicDataset(config)
    dataset.set_split(args.split)
    dataset.set_epoch(args.epoch)
    return dataset


def _aggregate_results(
    backend: str,
    workers: int,
    prefetch: int,
    batch_size: int,
    repeats: int,
    run_stats: List[Dict[str, Any]],
) -> BenchmarkResult:
    all_latencies: List[float] = []
    measured_samples = 0
    measured_batches = 0
    total_seconds = 0.0

    for stats in run_stats:
        all_latencies.extend(stats["latencies_ms"])
        measured_samples += int(stats["samples"])
        measured_batches += int(stats["batches"])
        total_seconds += float(stats["elapsed_s"])

    mean_ms = statistics.mean(all_latencies) if all_latencies else math.nan
    std_ms = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
    p50_ms = _safe_percentile(all_latencies, 50)
    p95_ms = _safe_percentile(all_latencies, 95)
    p99_ms = _safe_percentile(all_latencies, 99)
    min_ms = min(all_latencies) if all_latencies else math.nan
    max_ms = max(all_latencies) if all_latencies else math.nan
    batches_per_sec = measured_batches / total_seconds if total_seconds > 0 else 0.0
    samples_per_sec = measured_samples / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        backend=backend,
        workers=workers,
        prefetch=prefetch,
        batch_size=batch_size,
        repeats=repeats,
        measured_batches=measured_batches,
        measured_samples=measured_samples,
        mean_ms=mean_ms,
        std_ms=std_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        total_seconds=total_seconds,
        batches_per_sec=batches_per_sec,
        samples_per_sec=samples_per_sec,
    )


def print_summary(results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 132)
    print("MLX DATA PIPELINE BENCHMARK RESULTS")
    print("=" * 132)
    print(
        f"{'Backend':<12} {'Workers':>7} {'Prefetch':>8} {'Batch':>6} "
        f"{'Mean(ms)':>10} {'P50':>9} {'P95':>9} {'P99':>9} {'Std':>9} "
        f"{'Batches/s':>11} {'Samples/s':>11} {'Batches':>9} {'Samples':>10}"
    )
    print("-" * 132)

    sorted_results = sorted(
        results,
        key=lambda r: (-r.samples_per_sec, r.backend, r.workers, r.prefetch),
    )
    for r in sorted_results:
        print(
            f"{r.backend:<12} {r.workers:>7d} {r.prefetch:>8d} {r.batch_size:>6d} "
            f"{r.mean_ms:>10.2f} {r.p50_ms:>9.2f} {r.p95_ms:>9.2f} {r.p99_ms:>9.2f} {r.std_ms:>9.2f} "
            f"{r.batches_per_sec:>11.2f} {r.samples_per_sec:>11.2f} {r.measured_batches:>9d} {r.measured_samples:>10d}"
        )

    print("-" * 132)
    if sorted_results:
        best = sorted_results[0]
        print(
            "Best throughput: "
            f"{best.backend} workers={best.workers} prefetch={best.prefetch} "
            f"({best.samples_per_sec:.2f} samples/s, p95={best.p95_ms:.2f} ms)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX dynamic data pipeline")
    parser.add_argument("--speech-list", type=str, default=None, help="Speech file list (required without --cache-dir)")
    parser.add_argument("--noise-list", type=str, default=None, help="Noise file list (required without --cache-dir)")
    parser.add_argument("--rir-list", type=str, default=None, help="Optional RIR file list")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to cache dir from build_audio_cache.py")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--epoch", type=int, default=0, help="Dataset epoch seed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=200, help="Batches measured per repeat")
    parser.add_argument("--warmup-batches", type=int, default=10, help="Warmup batches per repeat")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per configuration")
    parser.add_argument(
        "--workers",
        type=parse_worker_list,
        default=parse_worker_list("1,2,4,8"),
        help="Comma-separated worker counts, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--backends",
        type=parse_backend_list,
        default=parse_backend_list("prefetch,mlx_stream"),
        help="Comma-separated backends: prefetch,mlx_stream",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch queue size for PrefetchDataLoader")
    parser.add_argument("--prefetch-size", type=int, default=8, help="Prefetch size for MLXDataStream")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--segment-length", type=float, default=5.0)
    parser.add_argument("--fft-size", type=int, default=960)
    parser.add_argument("--hop-size", type=int, default=480)
    parser.add_argument("--nb-erb", type=int, default=32)
    parser.add_argument("--nb-df", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not call mx.eval() on each batch (measures Python-side loader only)",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    dataset = _build_dataset(args)
    available_batches = len(dataset) // args.batch_size
    print(
        f"Split={args.split} samples={len(dataset):,} batch_size={args.batch_size} available_batches={available_batches:,}"
    )
    print(f"Measured batches/repeat={args.batches} warmup={args.warmup_batches} repeats={args.repeats}")
    print(f"Backends={args.backends} workers={args.workers}")
    if "mlx_stream" in args.backends and not HAS_MLX_DATA:
        print("Skipping mlx_stream backend: mlx-data is not installed.")

    sync_arrays = not args.no_sync
    results: List[BenchmarkResult] = []

    for backend in args.backends:
        if backend == "mlx_stream" and not HAS_MLX_DATA:
            continue

        for workers in args.workers:
            run_stats: List[Dict[str, Any]] = []
            prefetch = args.prefetch_factor if backend == "prefetch" else args.prefetch_size
            print(f"\nBenchmarking backend={backend} workers={workers} prefetch={prefetch}...")

            for run in range(args.repeats):
                dataset.set_split(args.split)
                dataset.set_epoch(args.epoch + run)

                if backend == "prefetch":
                    loader: Iterable[Dict[str, mx.array]] = PrefetchDataLoader(
                        dataset=dataset,
                        batch_size=args.batch_size,
                        num_workers=workers,
                        prefetch_factor=args.prefetch_factor,
                        drop_last=True,
                    )
                else:
                    stream = MLXDataStream(
                        dataset=dataset,
                        batch_size=args.batch_size,
                        prefetch_size=args.prefetch_size,
                        num_workers=workers,
                        drop_last=True,
                    )
                    stream.set_split(args.split)
                    stream.set_epoch(args.epoch + run)
                    loader = stream

                stats = _benchmark_loader_once(
                    loader=loader,
                    warmup_batches=args.warmup_batches,
                    measured_batches=args.batches,
                    sync_arrays=sync_arrays,
                )
                run_stats.append(stats)
                print(
                    f"  repeat {run + 1}/{args.repeats}: batches={stats['batches']} "
                    f"samples={stats['samples']} elapsed={stats['elapsed_s']:.2f}s"
                )
                if stats["batches"] < args.batches:
                    print(
                        "  warning: dataset exhausted before requested measured batches. "
                        "Increase dataset size or lower --batches."
                    )

            results.append(
                _aggregate_results(
                    backend=backend,
                    workers=workers,
                    prefetch=prefetch,
                    batch_size=args.batch_size,
                    repeats=args.repeats,
                    run_stats=run_stats,
                )
            )

    if not results:
        raise RuntimeError("No benchmark results generated")

    print_summary(results)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nWrote JSON results to {output_path}")


if __name__ == "__main__":
    main()
