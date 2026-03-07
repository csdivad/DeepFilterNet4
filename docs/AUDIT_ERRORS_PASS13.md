# Audit Errors Pass 13 — Dynamic Training Follow-up Closure

**Date:** 2026-03-06  
**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py`, `training_helpers.py`, `training_metrics.py`, run-profile support, and audit follow-up regressions  
**Baseline:** prior full-suite verification at 1022 passed / 11 skipped, plus targeted audit regressions rerun for this pass  
**Method:** control-flow tracing, checkpoint/resume semantics review, accumulation-window audit, GAN discriminator-path parity review, targeted regression execution

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| HIGH     | 0     | —     |
| MEDIUM   | 3     | 3     |
| LOW      | 1     | 1     |
| INFO     | 2     | 2     |
| **Total**| **6** | **6** |

**Verdict: PASS** — the follow-up audit surfaced four real behavioral defects in the dynamic training loop / metric path and verified two supporting workstreams (validation split isolation and DFN3 baseline profile tooling). All findings are now fixed and covered by regression tests.

---

## Findings

### ERR-13.1: Resumed `batch_idx` persisted post-resume local count instead of cumulative epoch progress — MEDIUM

**Files:** `DeepFilterNet/df_mlx/train_dynamic.py`, `DeepFilterNet/df_mlx/training_helpers.py`  
**Evidence:** On mid-epoch resume, `_update_interrupt_state(... batch_idx=num_train_batches ...)` wrote only the number of micro-batches processed *after* resuming. The persisted checkpoint/data position therefore moved backward relative to the already-consumed resume offset.

**Impact:** A second interruption in the same epoch could replay already-consumed data or desynchronize model/data checkpoints.

**Fix:** Added `completed_micro_batches(resume_batches_for_epoch, num_train_batches)` helper and used it consistently for interrupt-state and step-checkpoint metadata.

**Status:** FIXED

### ERR-13.2: End-of-epoch partial gradient-accumulation window was dropped — MEDIUM

**Files:** `DeepFilterNet/df_mlx/train_dynamic.py`, `DeepFilterNet/df_mlx/training_helpers.py`  
**Evidence:** The update path only flushed when `micro_batches_in_accum >= grad_accumulation_steps`. If an epoch ended with a remainder window (for example 10 micro-batches with accumulation=3), the final 1 micro-batch worth of gradients was discarded.

**Impact:** Silent loss of training signal and incorrect optimizer-step bookkeeping for scheduler progress.

**Fix:** Added `should_flush_grad_accumulation(..., is_last_micro_batch=...)` and scaled the final update by the actual remainder window size.

**Status:** FIXED

### ERR-13.3: Step checkpoints could fire on non-update micro-batches — MEDIUM

**Files:** `DeepFilterNet/df_mlx/train_dynamic.py`, `DeepFilterNet/df_mlx/training_helpers.py`  
**Evidence:** Save-by-steps was previously keyed only to `loop_state.global_step % save_steps == 0`, even on iterations where no optimizer update happened yet.

**Impact:** Duplicate or misleading step checkpoints, including the possibility of a save attempt at `global_step == 0`.

**Fix:** Added `should_save_step_checkpoint(...)` so save-by-steps requires a real optimizer update and a positive `global_step`.

**Status:** FIXED

### ERR-13.4: Sync-window GAN metrics used full-length discriminator inputs — LOW

**Files:** `DeepFilterNet/df_mlx/training_metrics.py`, `DeepFilterNet/tests/test_gan_memory_path.py`  
**Evidence:** The training path cropped GAN discriminator inputs with `_disc_crop_waveform(...)`, but sync-window metric collection used uncropped waveforms before the discriminator forward.

**Impact:** Avoidable memory/compute skew between the training path and the logging/metrics path.

**Fix:** Reused `_disc_crop_waveform(...)` in `collect_sync_metrics(...)` before discriminator inference.

**Status:** FIXED

### ERR-13.5: Validation split overflow regression remains fixed — INFO

**Files:** `DeepFilterNet/df_mlx/train_dynamic.py`, `DeepFilterNet/tests/test_validation_dataset_isolation.py`  
**Evidence:** Train/validation now use separate `DynamicDataset` instances, and the dedicated regression guard asserts the isolated validation dataset wiring.

**Status:** VERIFIED

### ERR-13.6: DFN3 baseline profile / ablation runner artifacts are present and loadable — INFO

**Files:** `DeepFilterNet/df_mlx/configs/run_profiles/baseline_dfn3_gan_vad_speech_full_vadlite.toml`, `DeepFilterNet/df_mlx/run_ablation_sweep.py`, associated tests  
**Evidence:** Added targeted tests verifying the full-run baseline profile semantics and the dry-run behavior of the resumable ablation sweep helper.

**Status:** VERIFIED

---

## Verification

### Targeted regression suite

```text
.venv/bin/python -m pytest \
  DeepFilterNet/tests/test_train_control_semantics.py \
  DeepFilterNet/tests/test_gan_memory_path.py \
  DeepFilterNet/tests/test_validation_dataset_isolation.py \
  DeepFilterNet/tests/test_dfn3_baseline_profiles.py \
  DeepFilterNet/tests/test_run_ablation_sweep.py -q

32 passed in 0.24s
```

### Prior full-suite verification used as audit baseline

```text
1022 passed, 11 skipped
```

---

## Files Changed in This Pass

| File | Purpose |
|------|---------|
| `DeepFilterNet/df_mlx/train_dynamic.py` | Resume counter, accumulation flush, step checkpoint gating |
| `DeepFilterNet/df_mlx/training_helpers.py` | Shared helpers for trainer counter/flush/save semantics |
| `DeepFilterNet/df_mlx/training_metrics.py` | Crop GAN metric waveforms before discriminator |
| `DeepFilterNet/tests/test_train_control_semantics.py` | Behavioral regressions for trainer control-flow fixes |
| `DeepFilterNet/tests/test_gan_memory_path.py` | Behavioral regression for GAN metric cropping |
| `DeepFilterNet/tests/test_dfn3_baseline_profiles.py` | DFN3 baseline profile verification |
| `DeepFilterNet/tests/test_run_ablation_sweep.py` | Resumable ablation runner verification |

---

## Conclusion

Pass 13 closes the follow-up audit loop for the dynamic training pipeline:

- checkpoint resume metadata is consistent across repeated interruptions,
- gradient accumulation no longer drops remainder windows,
- step checkpoints align with real optimizer updates,
- GAN sync metrics now match the training path’s memory discipline,
- and the DFN3 baseline profile / ablation tooling introduced during the same audit window is covered by executable tests.

No outstanding correctness blockers remain for the audited `df_mlx` training path.
