# Audit Errors Pass 12 — Runtime Error & Root-Cause Focus

**Date:** 2026-03-02  
**Scope:** `_audio_io.py`, `train_with_data.py`, `dynamic_dataset.py`, `prepare_data.py`, `training_losses.py`, `training_metrics.py`, `train_dynamic_config.py`, `grad_utils.py`, `augment_ext.py`  
**Method:** Signature tracing, fallback-path analysis, exception-handling audit, numeric guard review  
**Baseline:** 1007 tests passing (bc59082, main)

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| BLOCKER  | 0     | —     |
| HIGH     | 1     | 1     |
| MEDIUM   | 3     | 3     |
| LOW      | 3     | 0     |
| INFO     | 3     | 0     |
| **Total**| **10**| **4** |

**Verdict: CONDITIONAL-PASS** — 1 HIGH and 3 MEDIUM fixed; 3 LOW accepted as tolerable risk.

---

## Findings

### ERR-1: `_audio_io.py` scipy fallback missing uint8/float normalization — HIGH

**File:** `DeepFilterNet/df_mlx/_audio_io.py:39-46`  
**Evidence:** The scipy `wavfile.read` fallback normalizes `int16` (÷32768) and `int32` (÷2G) but has no handling for:
- `uint8` WAV files (common in legacy datasets) — scipy returns `[0, 255]`, must map to `[-1, 1]`
- `float32`/`float64` WAV files — scipy returns raw floats, which are already normalized but the code calls `.astype(np.float32)` on them, silently passing through `float64` precision loss (acceptable) but missing the `uint8` case entirely

**Root cause:** Incomplete dtype dispatch in the fallback branch. The `soundfile` primary path handles all formats via `dtype="float32"`, but the scipy fallback only handles the two most common integer formats.

**Runtime impact:** A uint8 WAV file loaded via the scipy fallback produces audio in `[0.0, 255.0]` range instead of `[-1.0, 1.0]`. This corrupts all downstream SNR calculations, feature extraction, and model training silently — no error raised, just wrong values.

**Fix:** Add `uint8` and `float64` normalization branches.  
**Status:** FIXED

---

### ERR-2: `_audio_io.py` empty file returns zero-length array — MEDIUM

**File:** `DeepFilterNet/df_mlx/_audio_io.py:25-32` (sf path), `39-46` (scipy path)  
**Evidence:** Both paths successfully read files that exist but contain zero audio samples. `sf.read()` returns `(array([]), sr)` and `wavfile.read()` returns `(sr, array([]))`. The zero-length array passes through mono conversion and resampling (which produces `int(0 * sr / file_sr) = 0` samples) and returns a zero-length `np.float32` array.

**Root cause:** No minimum-length validation after loading.

**Runtime impact:** Downstream consumers (`dynamic_dataset.py:_load_speech`, `prepare_data.py:process_file`) have length checks (`len(audio) < segment_samples`) that would filter these out. However, `load_audio_file` is a public API — direct callers could get zero-length arrays that crash on STFT computation (`compute_stft` with 0-length input → empty spectrogram → shape mismatch in batch assembly).

**Fix:** Add zero-length guard returning empty array warning in `load_audio_file_safe`, and an explicit ValueError in `load_audio_file`.  
**Status:** FIXED

---

### ERR-3: `prepare_data.py:270` silently swallows all exceptions in `process_file` — MEDIUM

**File:** `DeepFilterNet/df_mlx/prepare_data.py:268-272`  
**Evidence:**
```python
    except Exception:
        # Silently skip failed files in threaded mode
        pass
```
This is the outer `try/except` wrapping the entire `process_file` function. It catches *all* exceptions including `KeyboardInterrupt` (via `Exception` inheritance in some contexts), `MemoryError`, `SystemExit`, and importantly: **numpy errors from corrupt audio data** that would otherwise indicate dataset problems.

**Root cause:** Overly broad exception handling to support threaded execution.

**Runtime impact:** In production data preparation, entire files are silently dropped with no logging. If a large fraction of the dataset has encoding issues (e.g., wrong sample rate metadata), the resulting datastore could be much smaller than expected with no diagnostic output.

**Fix:** Narrow to specific expected exceptions and add logging.  
**Status:** FIXED

---

### ERR-4: `train_with_data.py` passes `best_valid_loss` as `**extra_state` but `save_checkpoint` stores it in JSON — MEDIUM

**File:** `DeepFilterNet/df_mlx/train_with_data.py:283-290` → `DeepFilterNet/df_mlx/train.py:521-585`  
**Evidence:** The call sites pass:
```python
save_checkpoint(model, optimizer, path, epoch=epoch+1, loss=avg_valid_loss, best_valid_loss=best_valid_loss)
```

The `save_checkpoint` signature is `(model, optimizer, path, epoch=0, step=0, loss=0.0, **extra_state)`. The `best_valid_loss` lands in `**extra_state` and is serialized via `json.dump({"epoch": ..., "step": ..., "loss": ..., **extra_state})`.

On resume: `state.get("best_valid_loss", float("inf"))` — this works correctly because the JSON state file contains the key from `**extra_state`.

However, `step` is never passed by `train_with_data.py` callers, so `step` is always 0 in checkpoints. This means any code relying on `state["step"]` for learning rate schedule warmup would get wrong behavior.

**Root cause:** `train_with_data.py` doesn't track `global_step` for checkpoint metadata, though it does track it locally. The `step=0` default silently produces wrong checkpoint state.

**Runtime impact:** If a resumed run uses `step` from checkpoint state for LR schedule calculations, the schedule would restart from step 0 instead of the actual step. Currently `train_with_data.py` recalculates `global_step = start_epoch * steps_per_epoch` on resume, which is an approximation (exact only if every epoch had identical batch counts). The mismatch is cosmetic for now but could corrupt LR schedules on partial-epoch resume.

**Fix:** Pass `step=global_step` in all `save_checkpoint` calls.  
**Status:** FIXED

---

### ERR-5: `dynamic_dataset.py` bare `except Exception` in audio loading — LOW

**File:** `DeepFilterNet/df_mlx/dynamic_dataset.py:973, 995, 1008`  
**Evidence:** `_load_speech`, `_load_noise`, and `_load_rir` all have `except Exception: return None` (or fallback to synthetic noise). These are in the per-sample hot path of training.

**Root cause:** Designed for resilience — training shouldn't crash from one bad file.

**Runtime impact:** Silent data loss during training. However, this is intentional and the fallback behavior (skip sample, use synthetic noise) is reasonable. The `PrefetchDataLoader` tracks `samples_failed` counts and can raise in `strict_failures` mode. **Acceptable risk** given the existing failure-tracking infrastructure.

**Verdict:** LOW — no change needed; the strict_failures mode provides adequate coverage.

---

### ERR-6: `augment_ext.py` `_mix_audio_python` division by zero on silent audio — LOW

**File:** `DeepFilterNet/df_mlx/augment_ext.py:173-175`  
**Evidence:**
```python
clean_power = np.mean(clean_out**2) + 1e-10
noise_power = np.mean(noise**2) + 1e-10
target_noise_power = clean_power / (10 ** (snr_db / 10))
```

The `1e-10` epsilon guards prevent division by zero. However, when `snr_db` is very large (e.g., +100 dB), `10 ** (100/10) = 1e10`, so `target_noise_power ≈ 1e-10 / 1e10 = 1e-20`, and `mix_factor = sqrt(1e-20 / 1e-10) ≈ 1e-5`. This is numerically fine.

When `snr_db` is very negative (e.g., -100 dB), `10 ** (-10) = 1e-10`, and `target_noise_power = 1e-10 / 1e-10 = 1.0`. This scales noise to match clean signal power, which is the intended behavior.

**Root cause:** Epsilon guards are present and adequate.

**Runtime impact:** None for realistic SNR ranges [-30, +40]. Extreme values (|SNR| > 100) could cause float underflow but this is outside operational bounds.

**Verdict:** LOW — pass; epsilon guards are sufficient for operational range.

---

### ERR-7: `training_losses.py` numeric guards are comprehensive — INFO (PASS)

**File:** `DeepFilterNet/df_mlx/training_losses.py:83-84, 155-159, 208, 302-303, 349, 432-435, 855`

**Evidence:** All `log10()` and `log()` calls have `+ eps` guards where `eps = 1e-8`:
- `mx.log10(clean_band + eps)` — guarded
- `mx.log(mag + eps)` — guarded
- Variance denominators use `mx.maximum(variance, _MIN_VARIANCE)` where `_MIN_VARIANCE = 1e-4` — prevents div-by-zero on silence
- Z-score denominators use `(sigma + eps)` — double-guarded (sigma already has min-variance floor)
- VAD slope uses `max(vad_z_slope, 1e-3)` — prevents div-by-zero from config
- Band bins denominators use `(band_bins + eps)` — guarded
- Speech ratios use `(clean_band + noise_band + eps)` — guarded

**Runtime impact:** None. All numeric paths are properly guarded against zero/negative log inputs and zero denominators.

**Verdict:** INFO — PASS. Numeric guards are thorough and consistent.

---

### ERR-8: `train_dynamic_config.py` `.get().lower()` safety — INFO (PASS)

**File:** `DeepFilterNet/df_mlx/train_dynamic_config.py:271, 279, 380`

**Evidence:** The `.get("key").lower()` calls at lines 271, 279, 380 are all inside `if "key" in sec:` guards. ConfigParser's `SectionProxy.get()` with a key known to exist always returns a `str`, never `None`. The Pyright issue is a type-annotation limitation — `SectionProxy.get()` has an overloaded signature that includes an `Optional[str]` fallback path, but the `in` check guarantees the key exists.

**Runtime impact:** None. These cannot return `None` at runtime.

**Verdict:** INFO — PASS. Confirmed non-issue (consistent with Pass 1 classification).

---

### ERR-9: `grad_utils.py` gradient leaf type safety — INFO (PASS)

**File:** `DeepFilterNet/df_mlx/grad_utils.py:29`

**Evidence:** `tree_flatten(grads)` returns `list[tuple[str, mx.array]]`. The `g * g` operation at line 29 operates on `mx.array` values. Pyright flags the `*` operator because `tree_flatten`'s return type annotation includes `Any` for the value type, but MLX gradient trees are always `mx.array` at runtime.

The non-finite gradient handling at lines 31-36 (`mx.isfinite`, `mx.where`, `mx.zeros_like`) correctly zeros all gradients when the norm is inf/nan — this is a robust safety net.

**Runtime impact:** None.

**Verdict:** INFO — PASS.

---

### ERR-10: `dynamic_dataset.py` `_load_speech` uses sample index as `rng` key — LOW

**File:** `DeepFilterNet/df_mlx/dynamic_dataset.py:1033`

**Evidence:**
```python
sample_seed = self.config.seed + self._epoch * 1000000 + idx
```

If `len(self._indices) > 1000000`, epoch seed spaces overlap. With typical datasets of 10k-100k files this is fine, but mega-scale datasets (>1M samples) would get seed collisions between epochs.

**Runtime impact:** Repeated noise/SNR pairings across epochs for large datasets. Not a correctness bug — just reduced diversity.

**Verdict:** LOW — acceptable for current scale; document as known limitation.

---

## Changes Made

### ERR-1 Fix: uint8/float normalization in scipy fallback

**File:** `DeepFilterNet/df_mlx/_audio_io.py`
- Added `uint8` normalization: `(audio.astype(np.float32) - 128.0) / 128.0`
- Added `float64` cast to `float32`
- No-op for `float32` (already correct)

### ERR-2 Fix: zero-length audio guard

**File:** `DeepFilterNet/df_mlx/_audio_io.py`
- `load_audio_file` raises `ValueError` on zero-length result
- `load_audio_file_safe` catches this and returns `None`

### ERR-3 Fix: narrower exception handling in `prepare_data.py`

**File:** `DeepFilterNet/df_mlx/prepare_data.py:process_file`
- Narrowed from `except Exception` to `except (OSError, ValueError, RuntimeError)`
- Added logging of skipped files with count

### ERR-4 Fix: pass `step=global_step` in checkpoint calls

**File:** `DeepFilterNet/df_mlx/train_with_data.py`
- All 3 `save_checkpoint` call sites now pass `step=global_step`

---

## Residual Risk

1. **ERR-5/6/10** (LOW): Accepted — guarded by existing fallback infrastructure; no remediation needed.
2. **ERR-7/8/9** (INFO): Confirmed non-issues — no action required.
3. No P0/P1 blockers found.

---

## Test Verification

All fixes verified against 1007-test baseline. See commit message for details.
