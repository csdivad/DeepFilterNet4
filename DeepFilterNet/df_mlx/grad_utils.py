"""Gradient-tree utilities shared across training/benchmark entrypoints."""

from __future__ import annotations

from typing import Any, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map


def clip_grad_norm_tree(grads: Any, max_norm: float) -> Tuple[Any, mx.array]:
    """Clip a gradient tree by global norm.

    Returns:
        (clipped_tree, total_norm_array)
    """
    leaves = tree_flatten(grads)
    if not leaves:
        return grads, mx.array(0.0)

    total_norm_sq = mx.array(0.0)
    for _, g in leaves:
        total_norm_sq = total_norm_sq + mx.sum(g * g)
    total_norm = mx.sqrt(total_norm_sq)
    # When total_norm is non-finite (inf/nan from exploding grads),
    # zero all gradients rather than propagating nan through the model.
    norm_finite = mx.isfinite(total_norm)
    safe_norm = mx.where(norm_finite, total_norm, mx.array(1.0))
    clip_coef = mx.minimum(max_norm / (safe_norm + 1e-6), mx.array(1.0))

    def _clip_leaf(g: mx.array) -> mx.array:
        clipped = g * clip_coef
        return mx.where(norm_finite, clipped, mx.zeros_like(g))

    return tree_map(_clip_leaf, grads), total_norm
