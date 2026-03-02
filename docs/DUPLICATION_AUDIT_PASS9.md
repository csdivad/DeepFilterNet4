# Duplication Audit — Pass 9

**Scope:** `DeepFilterNet/df_mlx/` module (intra-package duplication)  
**Branch:** `feat/final-audit`  
**Baseline:** 1007 tests pass, 11 skipped  
**Post-fix:** 1007 tests pass, 11 skipped  
**Date:** 2026-03-02

## Findings Summary

| ID | Sev | Component | Classification | Status |
|----|-----|-----------|----------------|--------|
| DUP-9.1 | P1 | train_with_data.py checkpoint I/O | Unnecessary | **FIXED** |
| DUP-9.2 | P2 | ROADMAP.md stale train_gan.py refs | Unnecessary | **FIXED** |
| DUP-9.3 | — | enhance.py / utils.py (load_audio/save_audio) | Justified | Keep |
| DUP-9.4 | — | train.py (WarmupCosineSchedule) / lr.py (CosineScheduler) | Justified | Keep |
| DUP-9.5 | — | 4× checkpoint implementations | Justified (3 of 4) | Keep |
| DUP-9.6 | — | df/ vs df_mlx/ cross-package | Justified (platform) | Keep |
| DUP-9.7 | — | MultiResolutionSTFTLoss delegation to SpectralLoss | Not duplication | Keep |
| DUP-9.8 | — | spectral_loss function vs SpectralLoss class | Justified (different input domains) | Keep |
| DUP-9.9 | P1 | load_audio_file copy-paste across 3 data scripts | Unnecessary | **FIXED** |

---

## Implemented Consolidations

### DUP-9.1 (P1): train_with_data.py checkpoint I/O → train.py

- **Files:** `df_mlx/train_with_data.py`
- **Evidence:** `save_checkpoint` (L47–92) and `load_checkpoint` (L95–131) in
  `train_with_data.py` reimplemented the same atomic-write checkpoint pattern
  that `train.py::save_checkpoint` (L521–584) and `train.py::load_checkpoint`
  (L588–630) already provide. The only differences were:
  - State file extension: `.state.json` → `.json` (merged to `.json`)
  - Extra param `best_valid_loss` → now flows through `train.py`'s `**extra_state`
  - Manual tree_flatten/tree_unflatten load → now uses `model.load_weights()`
- **Classification:** Unnecessary — no external callers imported these functions.
  The only test import from `train_with_data.py` is `clip_grad_norm` (L6 of
  `tests/test_grad_utils.py`), which is itself a re-export from `training_ops`.
- **Fix:** Removed inline definitions (~90 lines), replaced with:
  ```python
  from df_mlx.train import load_checkpoint, save_checkpoint
  ```
  Also removed dead `import json` and `import os`.
- **Call-site impact:** All 4 internal call sites (L124, L285, L312, L329) are
  compatible — positional args match, `best_valid_loss` flows through `**extra_state`.
- **Verification:** 1007 passed, 11 skipped (identical to baseline).
- **Commit:** `dfec862 refactor(train): delegate train_with_data.py checkpoint I/O to train.py (DUP-9.1)`

### DUP-9.2 (P2): ROADMAP.md stale train_gan.py references

- **Files:** `df_mlx/ROADMAP.md`
- **Evidence:** Two references to `train_gan.py` survived the file deletion
  (commit `2f31fe7`):
  - L61: Phase 2 task table referenced `train_gan.py` as file location
  - L155: Directory tree listed `train_gan.py` as "GAN training loop"
- **Fix:** Updated L61 to reference `train_dynamic.py` (which contains GAN
  training). Updated L155 to note GAN training was folded into `train_dynamic.py`.
- **Commit:** Same as DUP-9.1 (bundled).

---

## Justified Duplication (Keep)

### DUP-9.3: load_audio/save_audio in enhance.py vs utils.py

- **Files:** `df_mlx/enhance.py` (L88–174), `df_mlx/utils.py` (L22–85)
- **Rationale:** Different contracts:
  - `utils.py::load_audio` returns *target* sample rate; `enhance.py::load_audio`
    returns *original* sample rate.
  - `utils.py::save_audio` is minimal (3 args); `enhance.py::save_audio` has
    output_dir management, suffix appending, overwrite protection, and returns path.
  - Different resampling backends: `resampy` (utils) vs `scipy.signal` (enhance).
- **Risk if consolidated:** Breaking change for callers relying on different return
  values. Would require adding optional params that bloat the simple-case API.

### DUP-9.4: WarmupCosineSchedule (train.py) vs CosineScheduler+WarmupScheduler (lr.py)

- **Files:** `df_mlx/train.py` (L638–666), `df_mlx/lr.py` (L1–485)
- **Rationale:** Different complexity tiers:
  - `WarmupCosineSchedule` is a self-contained ~30-line callable used by examples,
    `train_dynamic.py`, `train_with_data.py`, and tests. Simple API, no state_dict.
  - `CosineScheduler` + `WarmupScheduler` is a full-featured scheduler system with
    `state_dict`/`load_state_dict`, cycle support, used by `checkpoint.py` and
    `__init__.py` re-exports.
  - 12+ callers of `WarmupCosineSchedule`, 13+ callers of `CosineScheduler`. High
    migration risk for negligible dedup gain.

### DUP-9.5: 4× checkpoint implementations

- **Files:** `checkpoint.py`, `training_checkpoints.py`, `train.py`, `train_with_data.py`
- **Rationale (for the 3 remaining):**
  - `train.py::save_checkpoint/load_checkpoint` — Standalone functions for examples
    and the `Trainer` class. Simple API (model + optional optimizer + path + metadata).
  - `checkpoint.py::save_checkpoint/load_checkpoint` — `CheckpointState`-based system
    with `PatienceState`, `CheckpointManager`, `CosineScheduler` integration. Used by
    `__init__.py` re-exports and `test_audit_fixes.py`.
  - `training_checkpoints.py::save_checkpoint/load_checkpoint` — Production training
    system with discriminator support, pipeline stages (gen-only → GAN), epoch markers,
    `validate_checkpoint_dir`. Used by `train_dynamic.py` and 6+ test files.
  - `train_with_data.py` — **Was** a 4th copy, now delegates to `train.py` (DUP-9.1).
- **Risk if consolidated further:** The three remaining implementations serve different
  layers of the system. `train.py` is the simple/example API; `checkpoint.py` is the
  public module API with patience tracking; `training_checkpoints.py` is the production
  training API with GAN support. Merging them would create a monolithic checkpoint
  module with parameters for all three use cases.

### DUP-9.6: df/ vs df_mlx/ cross-package

- **Files:** All overlapping modules (loss.py, checkpoint.py, enhance.py, utils.py,
  dnsmos_proxy.py, config.py, etc.)
- **Rationale:** `df/` is PyTorch; `df_mlx/` is Apple MLX. They share domain concepts
  but use fundamentally different frameworks. Per user directive, all cross-package
  duplication is JUSTIFIED as platform-specific implementations. Confirmed:
  - `df/utils.py` (19 functions) and `df_mlx/utils.py` (16 functions) have zero
    overlapping function names.
  - `df/config.py` uses Rust bindings; `df_mlx/config.py` uses Python dataclasses.
  - `dnsmos_proxy.py` in both packages: `torch.nn.Module` vs `mlx.nn.Module`.

---

### DUP-9.7 (Not duplication): MultiResolutionSTFTLoss delegates to SpectralLoss

- **Files:** `df_mlx/train.py` (L386–L453)
- **Evidence:** `MultiResolutionSTFTLoss.__init__` creates `self._spectral_loss = SpectralLoss(...)`.
  The `__call__` method delegates to `self._spectral_loss`. This is composition, not duplication.
- **Classification:** Not duplication — proper delegation pattern.

### DUP-9.8 (Justified): spectral_loss function vs SpectralLoss class

- **Files:** `df_mlx/train.py::spectral_loss` (L347), `df_mlx/loss.py::SpectralLoss` (L100)
- **Rationale:** Different input domains:
  - `spectral_loss()` operates on **pre-computed spectrograms** (real, imag) tuples
  - `SpectralLoss` class operates on **waveforms** and computes STFTs internally
- **Risk if consolidated:** Would conflate two different abstraction levels.

### DUP-9.9 (P1, FIXED): load_audio_file copy-paste across 3 data scripts

- **Files:**
  - `df_mlx/dynamic_dataset.py` (L77–L102): `load_audio_file` with soundfile/wavfile fallback
  - `df_mlx/build_audio_cache.py` (L64–L80): `load_audio_file` with error handling
  - `df_mlx/prepare_data.py` (L44–L71): `load_audio` with soundfile/wavfile fallback
- **Evidence:** All three implemented the identical load→mono→resample→float32 pipeline using
  `scipy.signal.resample`. The `build_audio_cache` version added `Optional[np.ndarray]` return.
- **Classification:** Unnecessary — identical backend-agnostic utility duplicated 3x.
- **Fix:** Created `_audio_io.py` as single canonical implementation:
  - `load_audio_file(path, sr)` → `np.ndarray` (soundfile with wavfile fallback)
  - `load_audio_file_safe(path, sr)` → `Optional[np.ndarray]` (wraps with error handling)
  - Updated all 3 callers to import from `_audio_io`
  - ~65 lines of duplicate code removed
- **Verification:** 1007 passed, 11 skipped (identical to baseline).

---

## Additional Observations

### Test files in df_mlx/ (not collected by pytest)

Six test files exist under `df_mlx/` but are NOT collected by pytest since
`pyproject.toml` sets `testpaths = ["tests"]`:

- `test_mlx.py` (328L) — manual test script with `if __name__ == "__main__"`
- `test_mlx_comprehensive.py` — comprehensive test suite
- `test_dynamic_dataset_safety.py`
- `test_enhance_cli_safety.py`
- `test_train_dynamic_config.py`
- `test_train_dynamic_resume_skip.py`

**Recommendation:** Move to `tests/` or document as manual-only integration tests.
Not a duplication finding but an observation for test coverage hygiene.

### Residual dead code from train_gan.py deletion

The `.vscode/tasks.json` still contains a task for deleting `train_gan.py` (stale).
Not blocking; cosmetic cleanup.

---

## Metrics

| Metric | Value |
|--------|-------|
| Files scanned | ~60 (df_mlx/), ~30 (df/) |
| Duplication hotspots found | 9 |
| Classified unnecessary | 3 (fixed) |
| Classified justified | 5 (documented) |
| Not duplication | 1 (delegation pattern) |
| Lines removed | ~155 |
| Tests before | 1007 passed, 11 skipped |
| Tests after | 1007 passed, 11 skipped |
| Regressions | 0 |
