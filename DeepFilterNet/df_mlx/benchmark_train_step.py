#!/usr/bin/env python3
"""Benchmark df_mlx training-step throughput and latency tails.

This benchmark targets training efficiency only:
- one forward + backward + optimizer update per measured step
- no validation pass
- no extra metrics-only forward pass

It compares data backend + worker/prefetch choices while measuring:
- data wait latency (next batch fetch)
- compute latency (forward/backward/update + sync)
- end-to-end step latency
- steps/s and samples/s

Example:
    python -m df_mlx.benchmark_train_step \
        --cache-dir /path/to/audio_cache \
        --backends prefetch,mlx_stream \
        --workers 2,4,8 \
        --prefetch-factor 2,4,8 \
        --prefetch-size 8,16,32 \
        --batch-size 8 \
        --steps 80 \
        --warmup-steps 10 \
        --repeats 2
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from df_mlx.config import get_default_config
from df_mlx.dynamic_dataset import (
    HAS_MLX_DATA,
    DatasetConfig,
    DynamicDataset,
    MLXDataStream,
    PrefetchDataLoader,
    read_file_list,
)
from df_mlx.model import init_model
from df_mlx.train import spectral_loss


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark matrix configuration."""

    backend: str
    split: str
    epoch: int
    workers: int
    prefetch: int
    batch_size: int
    warmup_steps: int
    steps: int
    repeats: int
    compiled: bool
    model_variant: str
    learning_rate: float
    weight_decay: float
    grad_clip: float
    sample_rate: int
    segment_length: float
    fft_size: int
    hop_size: int
    nb_erb: int
    nb_df: int
    seed: int


@dataclass
class BenchmarkResult:
    """Aggregated benchmark metrics for one case."""

    backend: str
    split: str
    epoch: int
    workers: int
    prefetch: int
    batch_size: int
    warmup_steps: int
    steps_requested: int
    repeats: int
    compiled: bool
    model_variant: str
    learning_rate: float
    weight_decay: float
    grad_clip: float
    sample_rate: int
    segment_length: float
    fft_size: int
    hop_size: int
    nb_erb: int
    nb_df: int
    seed: int
    measured_steps: int
    measured_samples: int
    total_seconds: float
    data_mean_ms: float
    data_p95_ms: float
    data_p99_ms: float
    step_mean_ms: float
    step_p95_ms: float
    step_p99_ms: float
    total_mean_ms: float
    total_p95_ms: float
    total_p99_ms: float
    steps_per_sec: float
    samples_per_sec: float
    loss_mean: float
    loss_std: float
    loss_last: float


def _parse_csv_tokens(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_int_list(value: str) -> List[int]:
    values: List[int] = []
    for token in _parse_csv_tokens(value):
        values.append(int(token))
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_float_list(value: str) -> List[float]:
    values: List[float] = []
    for token in _parse_csv_tokens(value):
        values.append(float(token))
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_bool_list(value: str) -> List[bool]:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    values: List[bool] = []
    for token in _parse_csv_tokens(value):
        lowered = token.lower()
        if lowered in truthy:
            values.append(True)
        elif lowered in falsy:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean value '{token}'. Use one of {sorted(truthy | falsy)}")
    if not values:
        raise ValueError("At least one boolean value is required")
    return values


def parse_backend_list(value: str) -> List[str]:
    valid = {"prefetch", "mlx_stream"}
    backends = _parse_csv_tokens(value)
    if not backends:
        raise ValueError("At least one backend is required")
    invalid = [b for b in backends if b not in valid]
    if invalid:
        raise ValueError(f"Invalid backends: {invalid}. Valid values: {sorted(valid)}")
    return backends


def parse_split_list(value: str) -> List[str]:
    valid = {"train", "valid", "test"}
    splits = _parse_csv_tokens(value)
    if not splits:
        raise ValueError("At least one split is required")
    invalid = [s for s in splits if s not in valid]
    if invalid:
        raise ValueError(f"Invalid split values: {invalid}. Valid values: {sorted(valid)}")
    return splits


def _require_min(name: str, values: Sequence[int], minimum: int) -> None:
    invalid = [value for value in values if value < minimum]
    if invalid:
        raise ValueError(f"{name} must be >= {minimum}, got {invalid}")


def _require_positive_float(name: str, values: Sequence[float]) -> None:
    invalid = [value for value in values if value <= 0.0]
    if invalid:
        raise ValueError(f"{name} must be > 0, got {invalid}")


def _safe_percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(values, q))


def _build_cases(args: argparse.Namespace) -> List[BenchmarkCase]:
    _require_min("epoch", args.epoch, 0)
    _require_min("workers", args.workers, 1)
    _require_min("batch-size", args.batch_size, 1)
    _require_min("warmup-steps", args.warmup_steps, 0)
    _require_min("steps", args.steps, 1)
    _require_min("repeats", args.repeats, 1)
    _require_min("sample-rate", args.sample_rate, 1)
    _require_positive_float("segment-length", args.segment_length)
    _require_min("fft-size", args.fft_size, 1)
    _require_min("hop-size", args.hop_size, 1)
    _require_min("nb-erb", args.nb_erb, 1)
    _require_min("nb-df", args.nb_df, 1)
    _require_min("prefetch-factor", args.prefetch_factor, 1)
    _require_min("prefetch-size", args.prefetch_size, 1)
    _require_positive_float("learning-rate", args.learning_rate)

    if not args.backends:
        raise ValueError("At least one backend is required")
    if not args.split:
        raise ValueError("At least one split is required")

    cases: List[BenchmarkCase] = []
    base = itertools.product(
        args.backends,
        args.split,
        args.epoch,
        args.workers,
        args.batch_size,
        args.warmup_steps,
        args.steps,
        args.repeats,
        args.compiled,
        args.learning_rate,
        args.weight_decay,
        args.grad_clip,
        args.sample_rate,
        args.segment_length,
        args.fft_size,
        args.hop_size,
        args.nb_erb,
        args.nb_df,
        args.seed,
    )

    for (
        backend,
        split,
        epoch,
        workers,
        batch_size,
        warmup_steps,
        steps,
        repeats,
        compiled,
        learning_rate,
        weight_decay,
        grad_clip,
        sample_rate,
        segment_length,
        fft_size,
        hop_size,
        nb_erb,
        nb_df,
        seed,
    ) in base:
        prefetch_values = args.prefetch_factor if backend == "prefetch" else args.prefetch_size
        for prefetch in prefetch_values:
            cases.append(
                BenchmarkCase(
                    backend=backend,
                    split=split,
                    epoch=epoch,
                    workers=workers,
                    prefetch=prefetch,
                    batch_size=batch_size,
                    warmup_steps=warmup_steps,
                    steps=steps,
                    repeats=repeats,
                    compiled=compiled,
                    model_variant=args.model_variant,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                    sample_rate=sample_rate,
                    segment_length=segment_length,
                    fft_size=fft_size,
                    hop_size=hop_size,
                    nb_erb=nb_erb,
                    nb_df=nb_df,
                    seed=seed,
                )
            )

    return cases


def _load_source_lists(args: argparse.Namespace) -> Dict[str, List[str]]:
    if args.cache_dir is None and (args.speech_list is None or args.noise_list is None):
        raise ValueError("Provide either --cache-dir or both --speech-list and --noise-list")

    source_lists: Dict[str, List[str]] = {"speech": [], "noise": [], "rir": []}

    if args.speech_list:
        source_lists["speech"] = read_file_list(args.speech_list)
    if args.noise_list:
        source_lists["noise"] = read_file_list(args.noise_list)
    if args.rir_list:
        source_lists["rir"] = read_file_list(args.rir_list)
    return source_lists


def _build_dataset(args: argparse.Namespace, case: BenchmarkCase, source_lists: Dict[str, List[str]]) -> DynamicDataset:
    config = DatasetConfig(
        cache_dir=args.cache_dir,
        speech_files=source_lists["speech"],
        noise_files=source_lists["noise"],
        rir_files=source_lists["rir"],
        sample_rate=case.sample_rate,
        segment_length=case.segment_length,
        fft_size=case.fft_size,
        hop_size=case.hop_size,
        nb_erb=case.nb_erb,
        nb_df=case.nb_df,
        seed=case.seed,
    )
    dataset = DynamicDataset(config)
    dataset.set_split(case.split)
    dataset.set_epoch(case.epoch)
    return dataset


def _make_loader(case: BenchmarkCase, dataset: DynamicDataset, epoch: int) -> Iterable[Dict[str, mx.array]]:
    dataset.set_split(case.split)
    dataset.set_epoch(epoch)

    if case.backend == "prefetch":
        return PrefetchDataLoader(
            dataset=dataset,
            batch_size=case.batch_size,
            num_workers=case.workers,
            prefetch_factor=case.prefetch,
            drop_last=True,
        )

    stream = MLXDataStream(
        dataset=dataset,
        batch_size=case.batch_size,
        prefetch_size=case.prefetch,
        num_workers=case.workers,
        drop_last=True,
    )
    stream.set_split(case.split)
    stream.set_epoch(epoch)
    return stream


def _clip_grad_norm(grads: Any, max_norm: float) -> Tuple[Any, mx.array]:
    flat_grads: List[mx.array] = []

    def flatten(x: Any) -> None:
        if isinstance(x, mx.array):
            flat_grads.append(x.reshape(-1))
        elif isinstance(x, dict):
            for v in x.values():
                flatten(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                flatten(v)

    flatten(grads)
    if not flat_grads:
        return grads, mx.array(0.0)

    total_norm_sq = sum(mx.sum(g**2) for g in flat_grads)
    total_norm = mx.sqrt(total_norm_sq)
    clip_coef = mx.minimum(max_norm / (total_norm + 1e-6), mx.array(1.0))

    def apply_clip(x: Any) -> Any:
        if isinstance(x, mx.array):
            return x * clip_coef
        if isinstance(x, dict):
            return {k: apply_clip(v) for k, v in x.items()}
        if isinstance(x, list):
            return [apply_clip(v) for v in x]
        if isinstance(x, tuple):
            return tuple(apply_clip(v) for v in x)
        return x

    return apply_clip(grads), total_norm


def _build_train_step(case: BenchmarkCase):
    mx.random.seed(case.seed)
    np.random.seed(case.seed)

    model_config = get_default_config()
    model_config.audio.sr = case.sample_rate
    model_config.audio.fft_size = case.fft_size
    model_config.audio.hop_size = case.hop_size
    model_config.audio.nb_freqs = (case.fft_size // 2) + 1
    model_config.audio.n_freqs = model_config.audio.nb_freqs
    model_config.erb.nb_erb = case.nb_erb
    model_config.df.nb_df = case.nb_df

    model = init_model(config=model_config, variant=case.model_variant)  # type: ignore[arg-type]
    model.train()

    optimizer: optim.Optimizer
    if case.weight_decay > 0:
        optimizer = optim.AdamW(learning_rate=case.learning_rate, weight_decay=case.weight_decay)
    else:
        optimizer = optim.Adam(learning_rate=case.learning_rate)

    def loss_fn(
        model_obj: nn.Module,
        noisy_real: mx.array,
        noisy_imag: mx.array,
        feat_erb: mx.array,
        feat_spec: mx.array,
        clean_real: mx.array,
        clean_imag: mx.array,
    ) -> mx.array:
        out = model_obj((noisy_real, noisy_imag), feat_erb, feat_spec)
        return spectral_loss(out, (clean_real, clean_imag))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    if case.compiled:

        @partial(mx.compile, inputs=state, outputs=state)
        def step_fn(
            noisy_real: mx.array,
            noisy_imag: mx.array,
            feat_erb: mx.array,
            feat_spec: mx.array,
            clean_real: mx.array,
            clean_imag: mx.array,
        ) -> mx.array:
            loss, grads = loss_and_grad(model, noisy_real, noisy_imag, feat_erb, feat_spec, clean_real, clean_imag)
            if case.grad_clip > 0:
                grads, _ = _clip_grad_norm(grads, case.grad_clip)
            optimizer.update(model, grads)
            return loss

    else:

        def step_fn(
            noisy_real: mx.array,
            noisy_imag: mx.array,
            feat_erb: mx.array,
            feat_spec: mx.array,
            clean_real: mx.array,
            clean_imag: mx.array,
        ) -> mx.array:
            loss, grads = loss_and_grad(model, noisy_real, noisy_imag, feat_erb, feat_spec, clean_real, clean_imag)
            if case.grad_clip > 0:
                grads, _ = _clip_grad_norm(grads, case.grad_clip)
            optimizer.update(model, grads)
            return loss

    return model, optimizer, step_fn


def _batch_size_from_batch(batch: Dict[str, mx.array]) -> int:
    snr = batch.get("snr")
    if snr is not None:
        return int(snr.shape[0])
    first = next(iter(batch.values()))
    return int(first.shape[0])


def _step_args(batch: Dict[str, mx.array]) -> Tuple[mx.array, ...]:
    return (
        batch["noisy_real"],
        batch["noisy_imag"],
        batch["feat_erb"],
        batch["feat_spec"],
        batch["clean_real"],
        batch["clean_imag"],
    )


def _benchmark_once(
    loader: Iterable[Dict[str, mx.array]],
    warmup_steps: int,
    measured_steps: int,
    step_fn: Any,
) -> Dict[str, Any]:
    iterator = iter(loader)

    warmup_done = 0
    while warmup_done < warmup_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        loss = step_fn(*_step_args(batch))
        mx.eval(loss)
        warmup_done += 1

    data_latencies_ms: List[float] = []
    step_latencies_ms: List[float] = []
    total_latencies_ms: List[float] = []
    losses: List[float] = []

    steps = 0
    samples = 0
    t_start = time.perf_counter()
    while steps < measured_steps:
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        t1 = time.perf_counter()

        loss = step_fn(*_step_args(batch))
        mx.eval(loss)
        t2 = time.perf_counter()

        data_ms = (t1 - t0) * 1000.0
        step_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0
        data_latencies_ms.append(data_ms)
        step_latencies_ms.append(step_ms)
        total_latencies_ms.append(total_ms)

        losses.append(float(loss))
        samples += _batch_size_from_batch(batch)
        steps += 1

    elapsed = time.perf_counter() - t_start
    return {
        "data_latencies_ms": data_latencies_ms,
        "step_latencies_ms": step_latencies_ms,
        "total_latencies_ms": total_latencies_ms,
        "losses": losses,
        "steps": steps,
        "samples": samples,
        "elapsed_s": elapsed,
    }


def _aggregate(case: BenchmarkCase, runs: List[Dict[str, Any]]) -> BenchmarkResult:
    data_latencies: List[float] = []
    step_latencies: List[float] = []
    total_latencies: List[float] = []
    losses: List[float] = []
    measured_steps = 0
    measured_samples = 0
    total_seconds = 0.0

    for run in runs:
        data_latencies.extend(run["data_latencies_ms"])
        step_latencies.extend(run["step_latencies_ms"])
        total_latencies.extend(run["total_latencies_ms"])
        losses.extend(run["losses"])
        measured_steps += int(run["steps"])
        measured_samples += int(run["samples"])
        total_seconds += float(run["elapsed_s"])

    data_mean = statistics.mean(data_latencies) if data_latencies else math.nan
    step_mean = statistics.mean(step_latencies) if step_latencies else math.nan
    total_mean = statistics.mean(total_latencies) if total_latencies else math.nan
    loss_mean = statistics.mean(losses) if losses else math.nan
    loss_std = statistics.stdev(losses) if len(losses) > 1 else 0.0
    loss_last = losses[-1] if losses else math.nan

    steps_per_sec = measured_steps / total_seconds if total_seconds > 0 else 0.0
    samples_per_sec = measured_samples / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        backend=case.backend,
        split=case.split,
        epoch=case.epoch,
        workers=case.workers,
        prefetch=case.prefetch,
        batch_size=case.batch_size,
        warmup_steps=case.warmup_steps,
        steps_requested=case.steps,
        repeats=case.repeats,
        compiled=case.compiled,
        model_variant=case.model_variant,
        learning_rate=case.learning_rate,
        weight_decay=case.weight_decay,
        grad_clip=case.grad_clip,
        sample_rate=case.sample_rate,
        segment_length=case.segment_length,
        fft_size=case.fft_size,
        hop_size=case.hop_size,
        nb_erb=case.nb_erb,
        nb_df=case.nb_df,
        seed=case.seed,
        measured_steps=measured_steps,
        measured_samples=measured_samples,
        total_seconds=total_seconds,
        data_mean_ms=data_mean,
        data_p95_ms=_safe_percentile(data_latencies, 95),
        data_p99_ms=_safe_percentile(data_latencies, 99),
        step_mean_ms=step_mean,
        step_p95_ms=_safe_percentile(step_latencies, 95),
        step_p99_ms=_safe_percentile(step_latencies, 99),
        total_mean_ms=total_mean,
        total_p95_ms=_safe_percentile(total_latencies, 95),
        total_p99_ms=_safe_percentile(total_latencies, 99),
        steps_per_sec=steps_per_sec,
        samples_per_sec=samples_per_sec,
        loss_mean=loss_mean,
        loss_std=loss_std,
        loss_last=loss_last,
    )


def print_summary(results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 168)
    print("DF_MLX TRAIN-STEP BENCHMARK RESULTS")
    print("=" * 168)
    print(
        f"{'Backend':<11} {'Comp':<5} {'Workers':>7} {'Pref':>5} {'Batch':>5} "
        f"{'MeanD':>8} {'P95D':>8} {'MeanS':>8} {'P95S':>8} {'MeanT':>8} {'P95T':>8} "
        f"{'Steps/s':>9} {'Samples/s':>10}"
    )
    print("-" * 168)

    ranked = sorted(results, key=lambda r: (-r.samples_per_sec, r.total_p95_ms, r.total_mean_ms))
    for r in ranked:
        print(
            f"{r.backend:<11} {str(r.compiled):<5} {r.workers:>7d} {r.prefetch:>5d} {r.batch_size:>5d} "
            f"{r.data_mean_ms:>8.2f} {r.data_p95_ms:>8.2f} {r.step_mean_ms:>8.2f} {r.step_p95_ms:>8.2f} "
            f"{r.total_mean_ms:>8.2f} {r.total_p95_ms:>8.2f} {r.steps_per_sec:>9.2f} {r.samples_per_sec:>10.2f}"
        )
    print("-" * 168)

    if ranked:
        best = ranked[0]
        print(
            "Best throughput: "
            f"{best.backend} compiled={best.compiled} workers={best.workers} prefetch={best.prefetch} "
            f"batch={best.batch_size} ({best.samples_per_sec:.2f} samples/s, total_p95={best.total_p95_ms:.2f} ms)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark df_mlx training-step efficiency")
    parser.add_argument("--speech-list", type=str, default=None, help="Speech file list (required without --cache-dir)")
    parser.add_argument("--noise-list", type=str, default=None, help="Noise file list (required without --cache-dir)")
    parser.add_argument("--rir-list", type=str, default=None, help="Optional RIR file list")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to cache dir from build_audio_cache.py")
    parser.add_argument("--split", type=parse_split_list, default=parse_split_list("train"))
    parser.add_argument("--epoch", type=parse_int_list, default=parse_int_list("0"))
    parser.add_argument("--workers", type=parse_int_list, default=parse_int_list("2,4,8"))
    parser.add_argument("--backends", type=parse_backend_list, default=parse_backend_list("prefetch,mlx_stream"))
    parser.add_argument(
        "--prefetch-factor",
        type=parse_int_list,
        default=parse_int_list("2,4,8"),
        help="Prefetch sizes for PrefetchDataLoader backend",
    )
    parser.add_argument(
        "--prefetch-size",
        type=parse_int_list,
        default=parse_int_list("8,16,32"),
        help="Prefetch sizes for MLXDataStream backend",
    )
    parser.add_argument("--batch-size", type=parse_int_list, default=parse_int_list("8"))
    parser.add_argument("--warmup-steps", type=parse_int_list, default=parse_int_list("10"))
    parser.add_argument("--steps", type=parse_int_list, default=parse_int_list("80"))
    parser.add_argument("--repeats", type=parse_int_list, default=parse_int_list("2"))
    parser.add_argument(
        "--compiled",
        type=parse_bool_list,
        default=parse_bool_list("true"),
        help="Compile step function with mx.compile (true/false list)",
    )
    parser.add_argument("--model-variant", choices=["full", "lite"], default="full")
    parser.add_argument("--learning-rate", type=parse_float_list, default=parse_float_list("0.001"))
    parser.add_argument("--weight-decay", type=parse_float_list, default=parse_float_list("0.0"))
    parser.add_argument("--grad-clip", type=parse_float_list, default=parse_float_list("0.0"))
    parser.add_argument("--sample-rate", type=parse_int_list, default=parse_int_list("48000"))
    parser.add_argument("--segment-length", type=parse_float_list, default=parse_float_list("5.0"))
    parser.add_argument("--fft-size", type=parse_int_list, default=parse_int_list("960"))
    parser.add_argument("--hop-size", type=parse_int_list, default=parse_int_list("480"))
    parser.add_argument("--nb-erb", type=parse_int_list, default=parse_int_list("32"))
    parser.add_argument("--nb-df", type=parse_int_list, default=parse_int_list("96"))
    parser.add_argument("--seed", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if "mlx_stream" in args.backends and not HAS_MLX_DATA:
        print("Skipping mlx_stream backend: mlx-data is not installed.")

    source_lists = _load_source_lists(args)
    cases = _build_cases(args)
    if not cases:
        raise RuntimeError("No benchmark cases generated from argument matrix")

    print(f"Benchmark matrix size: {len(cases)} configuration(s)")
    print(
        f"Backends={args.backends} splits={args.split} workers={args.workers} "
        f"batch_size={args.batch_size} compiled={args.compiled}"
    )

    results: List[BenchmarkResult] = []
    for idx, case in enumerate(cases, start=1):
        if case.backend == "mlx_stream" and not HAS_MLX_DATA:
            continue

        dataset = _build_dataset(args, case, source_lists)
        available_steps = len(dataset) // case.batch_size
        print(
            f"\n[{idx}/{len(cases)}] backend={case.backend} compiled={case.compiled} "
            f"split={case.split} epoch={case.epoch} workers={case.workers} prefetch={case.prefetch} "
            f"batch={case.batch_size} warmup={case.warmup_steps} steps={case.steps} repeats={case.repeats} "
            f"lr={case.learning_rate} wd={case.weight_decay} clip={case.grad_clip} "
            f"available_steps={available_steps}"
        )

        model, optimizer, step_fn = _build_train_step(case)
        run_stats: List[Dict[str, Any]] = []

        for run in range(case.repeats):
            loader = _make_loader(case, dataset, case.epoch + run)
            stats = _benchmark_once(
                loader=loader,
                warmup_steps=case.warmup_steps,
                measured_steps=case.steps,
                step_fn=step_fn,
            )
            run_stats.append(stats)
            print(
                f"  repeat {run + 1}/{case.repeats}: steps={stats['steps']} "
                f"samples={stats['samples']} elapsed={stats['elapsed_s']:.2f}s"
            )
            if stats["steps"] < case.steps:
                print(
                    "  warning: loader exhausted before requested measured steps. "
                    "Increase dataset size or reduce --steps."
                )

        results.append(_aggregate(case=case, runs=run_stats))

        del model
        del optimizer
        gc.collect()

    if not results:
        raise RuntimeError("No benchmark results generated")

    print_summary(results)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nWrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
