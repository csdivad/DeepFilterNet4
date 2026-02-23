# Performance Audit — Pass 9

**Date:** 2025-07-12
**Branch:** `feat/df_mlx-vad-head-gating`
**Scope:** `DeepFilterNet/df_mlx/` — newly added VadHead, VAD gating, BCE loss across training + inference
**Baseline:** 977 passed, 11 skipped (pytest)
**Prior passes:** Pass 4 (cached weight scalars, O(1) scheduler), Pass 6 (redundant compiled graph ops), Pass 7 (dtype cast elimination, z-scoring dedup, vectorized loss accum)

## Summary

1 optimization implemented (PERF-9.1). Broader audit of training loop, streaming inference, and gradient utilities found no further high-impact targets — the hot paths are already well-optimized by prior passes and `mx.compile` CSE.

## Findings

### PERF-9.1 — Defer VadHead sigmoid to inference; fuse BCE with logits (P1) ✅

| Field | Value |
|-------|-------|
| **Component** | `model.py` — VadHead, DfNet4, StreamingDfNet4; `train_dynamic.py` — 3 loss paths |
| **Evidence** | Profiling (10-iteration warmup + 50 timed iterations, batch=4×1×128×481): VadHead forward 1.186ms including unnecessary sigmoid on training path; 3 manual inline BCE blocks each computing `sigmoid → clamp → manual log formula` (0.283ms each) |
| **Before** | VadHead MLP ends with `nn.Sigmoid()`; all 3 training paths (compiled, GAN, eager) compute BCE manually: `p_ref_expanded * mx.log(vad_prob + eps) + (1 - p_ref_expanded) * mx.log(1 - vad_prob + eps)` |
| **After** | VadHead returns logits (sigmoid removed from MLP); `nn.losses.binary_cross_entropy(logits, target, with_logits=True)` uses fused log-sum-exp kernel; sigmoid applied only at inference gating points (DfNet4.__call__, StreamingDfNet4._forward_with_state, compiled_frame) |
| **Metrics** | VadHead forward: 1.186ms → 0.850ms (**28% faster**); BCE loss: 0.283ms → 0.171ms (**40% faster**); endpoint-to-endpoint improvement ~0.45ms/batch |
| **Files changed** | `model.py` (VadHead, DfNet4, StreamingDfNet4), `train_dynamic.py` (3 BCE blocks + variable renames), `test_mlx_comprehensive.py` (assertion updated for logits) |
| **Risk** | Low — semantically identical; `sigmoid(logit)` value unchanged at inference; BCEWithLogits is numerically more stable than manual sigmoid+log |
| **Verification** | 977 passed, 11 skipped (post-optimization benchmark + full test suite) |
| **Commit** | `2e6fc00` |

### PERF-9.2 — Z-scoring redundancy across loss functions (P2) — SKIPPED

| Field | Value |
|-------|-------|
| **Component** | `training_losses.py` — `_compute_vad_loss` + `_compute_vad_reg_loss` |
| **Evidence** | When both `use_vad_loss` and `use_vad_train_reg` are active, `_z_score_clean_energy` is called twice with identical inputs. However, the compiled loss function (`loss_fn`) runs under `mx.compile`, which performs CSE and shares the computation graph nodes automatically. The eager validation path runs infrequently. |
| **Rationale** | Compiler already handles this. Explicit sharing would add API complexity for negligible runtime gain. |

### PERF-9.3 — Redundant residual computation in eager path (P2) — SKIPPED

| Field | Value |
|-------|-------|
| **Component** | `train_dynamic.py` — eager training path, lines 2326-2327 |
| **Evidence** | `residual` and `residual_by_sample` compute the same squared differences with different reductions. Could derive `residual = mx.mean(residual_by_sample)`. |
| **Rationale** | Only in the eager fallback path (not the compiled hot path). The redundant computation creates extra lazy graph nodes but the cost is O(B×T×F) ≈ 500K elements — negligible vs model forward/backward. |

### PERF-9.4 — mu/sigma recomputation in `_compute_vad_probs` (P2) — SKIPPED

| Field | Value |
|-------|-------|
| **Component** | `training_losses.py` — `_compute_vad_probs` |
| **Evidence** | When `_precomputed_z` is provided, `log_clean` is available but `mu` and `sigma` are recomputed. Could extend `_z_score_clean_energy` return tuple to include them. |
| **Rationale** | O(B×T) reduction ops (tiny). In compiled path, CSE shares the nodes. Extending the return tuple would increase API surface for ~microsecond savings. |

## Audited Areas (No Issues Found)

| Area | Status | Notes |
|------|--------|-------|
| Compiled training step | ✅ | `mx.compile(inputs=state, outputs=state)` correctly captures full fwd+bwd+update. Periodic sync via `epoch_eval_frequency`. |
| Gradient accumulation | ✅ | Uses cached `_scale_cache` for scale arrays. Single sync point per eval window. `_tree_all_finite` batches checks via `mx.concatenate`. |
| Streaming inference (`compiled_frame`) | ✅ | Fully compiled with `@mx.compile`. VAD gating adds 2 ops (sigmoid+max). Mamba states stacked via `mx.stack` (required by compile return contract). |
| `specs_to_wavs` / ISTFT | ✅ | Straightforward FFT calls, no overhead. FP32 casting is conditional. |
| `_compute_awesome_losses` | ✅ | FP32 cast at entry with `_assume_float32=True` downstream (PERF-7.1). Noise pre-computation shared. Proxy gates use `_precomputed_z` (PERF-7.2). |
| Discriminator update path | ✅ | `mx.stop_gradient` on cached waveforms before disc forward. `mx.clear_cache()` between gen and disc to reduce peak memory. |
| Compiled step retrace guards | ✅ | Batch-size mismatch falls back to eager with warning. No unnecessary retraces. |

## Conclusion

The VadHead + BCE optimization (PERF-9.1) was the only high-impact target in the newly added VAD code paths. The broader training loop is well-optimized by prior passes (4, 6, 7) and benefits from `mx.compile` graph optimization. No further implementation-worthy findings at this time.

**Test state:** 977 passed, 11 skipped (unchanged from baseline)
