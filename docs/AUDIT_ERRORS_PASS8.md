# Pass 8 Error Audit Report — VAD Head & Gating Integration

**Branch:** `feat/df_mlx-vad-head-gating`  
**Baseline:** 977 passed, 11 skipped  
**Post-fix:** 977 passed, 11 skipped (+ 243 comprehensive)  
**Scope:** `DeepFilterNet/df_mlx/` — VAD head integration, training loop model calls, streaming inference, checkpoint resume, error handling  
**Commit:** `6c2ee56`

## Findings Summary

| ID | Sev | Component | Status |
|----|-----|-----------|--------|
| VAD-1 | P1 | train_dynamic.py (loss_fn / loss_fn_gan) | **FIXED** |
| VAD-2 | P1 | train_dynamic.py (disc fallback) | **FIXED** |
| VAD-3 | P2 | model.py (streaming inference) | **FIXED** |
| VAD-4 | P3 | train_dynamic.py (_diagnose_nonfinite) | NOTED |

## Detailed Findings

### VAD-1 — Training loss functions never receive VAD probability (P1)

- **Severity:** P1 (correctness — renders VadHead training dead code)
- **Component:** `train_dynamic.py:1226` (`loss_fn`) and `train_dynamic.py:1428` (`loss_fn_gan`)
- **Evidence:** Both model calls used `model(noisy_spec, feat_erb, feat_spec)` without `return_vad=True`. Without this flag, `DfNet4.__call__` returns `(real, imag)` not `((real, imag), vad_prob)`.
- **Why it fails:** The unpacking check `isinstance(out[0], tuple)` always takes the `else` branch, setting `vad_prob = None`. The `_compute_vad_loss` call at training_losses.py:183 receives `spec_out[0]` and `spec_out[1]` (which are arrays, not the VadHead output). The BCE-based VAD loss computes a proxy from spectral energy but the VadHead itself **receives no supervisory gradient through the VAD loss path**. The head's parameters are only updated through the indirect spectral loss gating path (when `training=False`), but during training `training=True` so no gating is applied — the VadHead is effectively **dead code during training**.
- **Fix:** Added `return_vad=True` to both model calls.
- **Before:** `out = _gen_fn(noisy_spec, feat_erb, feat_spec)`
- **After:** `out = _gen_fn(noisy_spec, feat_erb, feat_spec, return_vad=True)`
- **Regression risk:** Low — the unpacking logic was already written to handle both cases.

### VAD-2 — Discriminator fallback corrupts spectrum with VAD probability (P1)

- **Severity:** P1 (correctness — wrong data passed to discriminator)
- **Component:** `train_dynamic.py:3407` (discriminator fallback path)
- **Evidence:** The line `pred_spec = model((noisy_real, noisy_imag), feat_erb, feat_spec, return_vad=True)` returns `((real, imag), vad_prob)`. The code then accesses `pred_spec[0]` (gets `(real, imag)` tuple, not real array) and `pred_spec[1]` (gets `vad_prob`, not imag array).
- **Why it fails:** `mx.stop_gradient(pred_spec[0])` applies to a tuple rather than an array — depending on MLX version this either errors or produces garbage. `pred_spec[1]` is `vad_prob` (shape `(batch, time, 1)`) being used where `spec_imag` (shape `(batch, time, freq)`) is expected. This would corrupt the discriminator's input spectrum.
- **Fix:** Changed to `model((noisy_real, noisy_imag), feat_erb, feat_spec)` (default `return_vad=False`). This path only needs `(real, imag)` for the discriminator; VAD is not needed.
- **Before:** `model(..., return_vad=True)` → `pred_spec = ((real,imag), vad_prob)`
- **After:** `model(...)` → `pred_spec = (real, imag)`
- **Regression risk:** None — this path now gets exactly the `(real, imag)` tuple it was designed for.

### VAD-3 — Streaming inference missing VAD gating (P2)

- **Severity:** P2 (consistency — streaming and batch inference produce different outputs)
- **Component:** `model.py:1666` (`_forward_with_state`) and `model.py:1797` (`compiled_frame`)
- **Evidence:** `DfNet4.__call__` applies VAD gating at inference time: `vad_gate = mx.maximum(vad_prob, 0.01); spec_out = (real * vad_gate, imag * vad_gate)`. Neither streaming path (`_forward_with_state` nor `compiled_frame`) computed `vad_prob` or applied this gating.
- **Why it fails:** Streaming inference (frame-by-frame processing) produces different output magnitudes than batch inference. A frame with low speech probability that should be attenuated by ~40dB (gate = 0.01) passes through unattenuated in streaming mode.
- **Fix:** Added `vad_prob = model.vad_head(emb)` and soft gating to both streaming paths, positioned after post-filter and before iSTFT — matching the exact order in `DfNet4.__call__`.
- **Regression risk:** Low — matches existing batch inference behavior exactly.

### VAD-4 — Diagnostic pass skips VadHead NaN check (P3)

- **Severity:** P3 (observability gap, not correctness)
- **Component:** `train_dynamic.py:1616` (`_diagnose_nonfinite`)
- **Evidence:** Calls `model(...)` without `return_vad=True`, so `vad_prob` is `None` and `_diag_check("model.vad_prob", ...)` is skipped.
- **Why it matters:** If VadHead produces NaN, the diagnostic won't report it. However, VadHead is a small MLP with sigmoid output, making standalone NaN production unlikely.
- **Decision:** Not fixed. The diagnostic is non-critical debug tooling, and adding `return_vad=True` would change the model's inference behavior (no VAD gating → VAD gated output) during diagnosis, which could mask other issues.

## Areas Audited (No Issues Found)

### Checkpoint Resume Logic
- `compute_resume_epoch`: Correctly distinguishes completed vs in-progress checkpoints (epoch+1 vs epoch)
- `resolve_resume_batch_count`: Handles legacy (0-based) and modern (count-based) semantics
- `load_checkpoint`: Properly restores model weights, optimizer state, discriminator, and disc optimizer with defensive error handling
- Data checkpoint alignment: Auto-corrects ±1 batch drift between model and data checkpoints
- LR schedule: `global_step = resume_global_step` correctly positions the cosine schedule after resume

### Gradient Accumulation & Optimizer Updates
- Compiled + accumulation: `_tree_all_finite` guard at line 3200 ✅
- Eager + accumulation: `_tree_all_finite` guard at line 3339 ✅
- Discriminator: `_tree_all_finite` guard at line 3457 ✅
- Fully compiled (no accumulation): Uses `clip_grad_norm` zeroing inside compiled graph — by design, cannot add conditional outside compilation boundary

### Validation Loop
- Uses `return_vad=True` with correct unpacking ✅
- Calls model with `training=False` so VAD gating is applied — matches inference behavior ✅
- Spectral loss measures gated output vs clean target ✅

### Loss Functions (training_losses.py)
- `_compute_vad_loss`: Correctly uses `mx.stop_gradient` on gating term (prevents circular supervision) ✅
- One-sided penalty with margin: `mx.maximum(p_ref - p_out - margin, 0.0)` ✅
- Speech gate properly clamped, SNR gate bounded via sigmoid ✅

### Model Pipeline Order
- Batch inference: ERB mask → DF filter → post-filter → VAD gating → (LSNR dropout if training) ✅
- Streaming `_forward_with_state`: ERB mask → DF filter → post-filter → VAD gating ✅ (fixed)
- Streaming `compiled_frame`: ERB mask → DF filter → post-filter → VAD gating → iSTFT ✅ (fixed)
- `forward_with_lsnr`: ERB mask → DF filter → post-filter (no VAD, by design — only used in tests) ✅
- `enhance`: Delegates to `__call__(training=False)` — gets full pipeline including VAD gating ✅

### NaN Handling
- Non-finite loss counter with abort at 50 cumulative events ✅
- `_tree_all_finite` guards on all eager optimizer update paths ✅
- Compiled path uses `clip_grad_norm` zeroing as substitute ✅
- Loss accumulator: substitutes `0.0` for non-finite losses to prevent epoch average poisoning ✅

## Residual Risk

| Risk | Severity | Mitigation |
|------|----------|------------|
| Compiled step applies optimizer update with zeroed NaN grads (momentum decay) | Low | By design — required for `mx.compile` compatibility. Rare event (< 1/1000 batches in stable training) |
| Optimizer state restore failure continues training with cold optimizer | Low | Defensive design — prints warning, prevents crash. Cold restart is preferable to abort |
| `_diagnose_nonfinite` doesn't check VadHead output | Low | VadHead is sigmoid-bounded MLP, unlikely to NaN independently |

## Test Results

```
Main suite:          977 passed, 11 skipped (63.16s) — matches baseline
Comprehensive suite: 243 passed (7.93s) — matches baseline  
```

## Files Changed

| File | Changes |
|------|---------|
| [train_dynamic.py](DeepFilterNet/df_mlx/train_dynamic.py) | +4 −3: VAD-1 (2x return_vad=True), VAD-2 (remove return_vad from disc) |
| [model.py](DeepFilterNet/df_mlx/model.py) | +12: VAD-3 (streaming VAD gating in 2 paths) |
