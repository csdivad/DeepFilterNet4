"""Numeric debugging, batch conversion, and gradient accumulation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from df_mlx.grad_utils import clip_grad_norm_tree


def _batch_to_float(*arrays: mx.array) -> tuple[float, ...]:
    """Evaluate multiple MLX arrays in one sync, then extract Python floats.

    Reduces N individual ``float(mx_array)`` sync barriers to a single ``mx.eval()``.
    """
    mx.eval(*arrays)
    return tuple(float(a) for a in arrays)


@dataclass
class NumericDebugConfig:
    enabled: bool = False
    fail_fast: bool = True
    skip_batch: bool = False
    every: int = 1
    dump_dir: Path | None = None
    dump_arrays: bool = False
    max_dumps: int = 5
    check_grads: bool = True


class NumericDebugger:
    """Helper for fail-fast finite checks and debug dumps."""

    def __init__(self, config: NumericDebugConfig):
        self.config = config
        self.dump_count = 0

    def _should_check(self, ctx: dict[str, Any] | None) -> bool:
        if not self.config.enabled:
            return False
        if ctx is None:
            return True
        step = ctx.get("global_step")
        if isinstance(step, int):
            return (step % max(self.config.every, 1)) == 0
        return True

    def _dump_stats(self, name: str, tensor: mx.array, ctx: dict[str, Any] | None) -> None:
        if self.config.dump_dir is None:
            return
        if self.dump_count >= self.config.max_dumps:
            return
        self.config.dump_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(tensor, dtype=np.float32)
        finite_mask = np.isfinite(arr)
        finite_vals = arr[finite_mask]
        if finite_vals.size > 0:
            stats = {
                "min": float(finite_vals.min()),
                "max": float(finite_vals.max()),
                "mean": float(finite_vals.mean()),
            }
        else:
            stats = {"min": None, "max": None, "mean": None}
        dump = {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_pct": float(100.0 * finite_mask.mean()),
            "nonfinite_count": int(arr.size - finite_mask.sum()),
            "stats": stats,
            "context": ctx or {},
        }
        out_path = self.config.dump_dir / f"nonfinite_{self.dump_count:03d}_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        if self.config.dump_arrays:
            slices = tuple(slice(0, min(dim, 8)) for dim in arr.shape)
            sample = arr[slices]
            np.savez_compressed(
                self.config.dump_dir / f"nonfinite_{self.dump_count:03d}_{name}.npz",
                sample=sample,
            )
        self.dump_count += 1

    def check(self, name: str, tensor: mx.array, ctx: dict[str, Any] | None = None) -> bool:
        if not self._should_check(ctx):
            return True
        is_finite = mx.isfinite(tensor)
        if bool(mx.all(is_finite)):
            return True
        self._dump_stats(name, tensor, ctx)
        message = f"Non-finite detected in {name}"
        if ctx:
            message += f" | ctx={ctx}"
        if self.config.fail_fast:
            raise FloatingPointError(message)
        return False

    def check_tree(self, name: str, tree: Any, ctx: dict[str, Any] | None = None) -> bool:
        """Check gradient tree for non-finite values.

        Never raises even when fail_fast is set — gradient non-finiteness is
        handled by skipping the optimizer update, not by crashing.  Dumps are
        still written for post-mortem analysis.
        """
        if not self._should_check(ctx) or not self.config.check_grads:
            return True

        all_finite = True
        for key, value in tree_flatten(tree):
            if value is None:
                continue
            if not bool(mx.all(mx.isfinite(value))):
                key_name = f"{name}.{key}"
                self._dump_stats(key_name, value, ctx)
                all_finite = False
        if not all_finite:
            from tqdm import tqdm

            tqdm.write(f"⚠️  Non-finite gradients in {name} " f"(ctx={ctx}) — skipping optimizer update")
        return all_finite


def _tree_all_finite(tree: Any) -> bool:
    """Fast tree-wide finite check (no dumps)."""
    for _, value in tree_flatten(tree):
        if value is None:
            continue
        if not bool(mx.all(mx.isfinite(value))):
            return False
    return True


def clip_grad_norm(grads, max_norm: float) -> Tuple[dict, mx.array]:
    """Clip gradients by global norm.

    Returns:
        Tuple of (clipped_grads, grad_norm) where grad_norm is an MLX array.
        Call float(grad_norm) outside compiled functions to get the scalar value.
    """
    clipped, total_norm = clip_grad_norm_tree(grads, max_norm)
    return cast(dict, clipped), total_norm


def accumulate_grads(accumulated: Any | None, new_grads: Any) -> Any:
    """Accumulate gradients by summing them element-wise.

    Args:
        accumulated: Previous accumulated gradients (None for first batch)
        new_grads: New gradients to add

    Returns:
        Combined gradient tree
    """
    if accumulated is None:
        return new_grads

    def add_trees(a: Any, b: Any) -> Any:
        if isinstance(a, mx.array) and isinstance(b, mx.array):
            return a + b
        elif isinstance(a, dict) and isinstance(b, dict):
            return {k: add_trees(a[k], b[k]) for k in a.keys()}
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            result = [add_trees(av, bv) for av, bv in zip(a, b)]
            return type(a)(result)
        return b  # fallback (shouldn't happen with valid grad trees)

    return add_trees(accumulated, new_grads)


def scale_grads(grads: Any, scale: float) -> Any:
    """Scale all gradients by a constant factor.

    Args:
        grads: Gradient tree
        scale: Scale factor (e.g., 1/grad_accumulation_steps)

    Returns:
        Scaled gradient tree
    """
    scale_arr = mx.array(scale, dtype=mx.float32)

    def apply_scale(x: Any) -> Any:
        if isinstance(x, mx.array):
            return x * scale_arr
        elif isinstance(x, dict):
            return {k: apply_scale(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [apply_scale(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(apply_scale(v) for v in x)
        return x

    return apply_scale(grads)
