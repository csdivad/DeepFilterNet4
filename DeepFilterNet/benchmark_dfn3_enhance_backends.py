"""Benchmark DeepFilterNet3 enhancement backends on Apple Silicon.

This script compares the end-to-end enhancement functions used by:
- ``df.enhance`` running the original PyTorch model on ``mps``
- ``df_mlx.enhance`` running the converted MLX model

Both paths use the shipped DeepFilterNet3 weights from this repository:
- PyTorch: ``models/DeepFilterNet3.zip``
- MLX: ``models/mlx/DeepFilterNet3-MLX``

The benchmark exercises the real enhancement functions on the same deterministic
waveform so performance claims are based on the full inference path rather than
isolated forward passes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from df.enhance import enhance as torch_enhance
from df.enhance import init_df as init_torch_df
from df_mlx.enhance import enhance as mlx_enhance
from df_mlx.enhance import load_model as load_mlx_model

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_ARCHIVE = REPO_ROOT / "models" / "DeepFilterNet3.zip"
MLX_MODEL_DIR = REPO_ROOT / "models" / "mlx" / "DeepFilterNet3-MLX"


@dataclass
class BackendResult:
    """Measured timings for one enhancement backend."""

    name: str
    warm_times_s: list[float]
    cold_time_s: float

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.warm_times_s)

    @property
    def std_s(self) -> float:
        if len(self.warm_times_s) < 2:
            return 0.0
        return statistics.stdev(self.warm_times_s)

    @property
    def p95_s(self) -> float:
        return float(np.percentile(self.warm_times_s, 95))


@dataclass
class BenchmarkReport:
    """Structured comparison report."""

    audio_seconds: float
    sample_rate: int
    warmup_runs: int
    benchmark_runs: int
    torch_mps: BackendResult
    mlx: BackendResult

    def to_json_dict(self) -> dict[str, Any]:
        data = {
            "audio_seconds": self.audio_seconds,
            "sample_rate": self.sample_rate,
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "torch_mps": asdict(self.torch_mps),
            "mlx": asdict(self.mlx),
            "speedup": {
                "steady_state_mlx_vs_torch": self.torch_mps.mean_s / self.mlx.mean_s,
                "cold_start_mlx_vs_torch": self.torch_mps.cold_time_s / self.mlx.cold_time_s,
            },
        }
        data["torch_mps"].update(
            {"mean_s": self.torch_mps.mean_s, "std_s": self.torch_mps.std_s, "p95_s": self.torch_mps.p95_s}
        )
        data["mlx"].update({"mean_s": self.mlx.mean_s, "std_s": self.mlx.std_s, "p95_s": self.mlx.p95_s})
        return data


def build_waveform(sample_rate: int, duration_s: float, seed: int) -> np.ndarray:
    """Create a deterministic speech-like synthetic waveform."""

    samples = int(sample_rate * duration_s)
    rng = np.random.default_rng(seed)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    waveform = (
        0.15 * np.sin(2 * np.pi * 220.0 * t)
        + 0.10 * np.sin(2 * np.pi * 440.0 * t)
        + 0.05 * np.sin(2 * np.pi * 880.0 * t)
        + 0.02 * rng.standard_normal(samples)
    )
    return waveform.astype(np.float32)


def _assert_prereqs() -> None:
    if not TORCH_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing PyTorch archive: {TORCH_ARCHIVE}")
    if not MLX_MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing MLX model directory: {MLX_MODEL_DIR}")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("PyTorch MPS backend is not available on this machine")


def _extract_torch_model_dir() -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="dfn3_torch_model_")
    with zipfile.ZipFile(TORCH_ARCHIVE) as archive:
        archive.extractall(temp_dir.name)
    return temp_dir


def benchmark_torch_mps(waveform: np.ndarray, warmup_runs: int, benchmark_runs: int) -> BackendResult:
    """Benchmark the original PyTorch ``df.enhance`` path on MPS."""

    model_dir_ctx = _extract_torch_model_dir()
    model_dir = Path(model_dir_ctx.name) / "DeepFilterNet3"
    audio = torch.from_numpy(waveform).unsqueeze(0)

    try:
        model, df_state, _, _ = init_torch_df(
            model_base_dir=str(model_dir),
            device="mps",
            log_level="ERROR",
            config_allow_defaults=True,
        )

        for _ in range(warmup_runs):
            enhanced = torch_enhance(model, df_state, audio, pad=True, atten_lim_db=None, device="mps")
            torch.mps.synchronize()
            _ = enhanced.shape

        warm_times: list[float] = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            enhanced = torch_enhance(model, df_state, audio, pad=True, atten_lim_db=None, device="mps")
            torch.mps.synchronize()
            _ = enhanced.shape
            warm_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model_cold, df_state_cold, _, _ = init_torch_df(
            model_base_dir=str(model_dir),
            device="mps",
            log_level="ERROR",
            config_allow_defaults=True,
        )
        enhanced = torch_enhance(model_cold, df_state_cold, audio, pad=True, atten_lim_db=None, device="mps")
        torch.mps.synchronize()
        _ = enhanced.shape
        cold_time = time.perf_counter() - start

        return BackendResult(name="PyTorch-MPS", warm_times_s=warm_times, cold_time_s=cold_time)
    finally:
        model_dir_ctx.cleanup()


def benchmark_mlx(waveform: np.ndarray, warmup_runs: int, benchmark_runs: int) -> BackendResult:
    """Benchmark the MLX ``df_mlx.enhance`` path."""

    audio = mx.array(waveform)
    model, params, _, _ = load_mlx_model(str(MLX_MODEL_DIR), epoch="latest")

    for _ in range(warmup_runs):
        enhanced = mlx_enhance(model, audio, params, compensate_delay=True, atten_lim_db=None)
        mx.eval(enhanced)

    warm_times: list[float] = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        enhanced = mlx_enhance(model, audio, params, compensate_delay=True, atten_lim_db=None)
        mx.eval(enhanced)
        warm_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    model_cold, params_cold, _, _ = load_mlx_model(str(MLX_MODEL_DIR), epoch="latest")
    enhanced = mlx_enhance(model_cold, audio, params_cold, compensate_delay=True, atten_lim_db=None)
    mx.eval(enhanced)
    cold_time = time.perf_counter() - start

    return BackendResult(name="MLX", warm_times_s=warm_times, cold_time_s=cold_time)


def run_benchmark(duration_s: float, warmup_runs: int, benchmark_runs: int, seed: int) -> BenchmarkReport:
    """Run both backend benchmarks and return a structured report."""

    _assert_prereqs()
    sample_rate = 48_000
    waveform = build_waveform(sample_rate, duration_s, seed)

    torch_result = benchmark_torch_mps(waveform, warmup_runs, benchmark_runs)
    mlx_result = benchmark_mlx(waveform, warmup_runs, benchmark_runs)
    return BenchmarkReport(
        audio_seconds=duration_s,
        sample_rate=sample_rate,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        torch_mps=torch_result,
        mlx=mlx_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="Synthetic waveform duration in seconds")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs per backend")
    parser.add_argument("--benchmark-runs", type=int, default=5, help="Number of timed runs per backend")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the waveform")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_benchmark(args.duration, args.warmup_runs, args.benchmark_runs, args.seed)
    payload = report.to_json_dict()
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
