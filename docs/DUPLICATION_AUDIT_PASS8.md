# Duplication Audit — Pass 8

**Scope:** `DeepFilterNet/df_mlx/` module  
**Branch:** `feat/df_mlx-vad-head-gating`  
**Baseline:** 977 tests pass, 11 skipped  
**Date:** 2025-07-15

## Findings Summary

| ID | Sev | Component | Classification | Status |
|----|-----|-----------|----------------|--------|
| DUP-8.1 | P2 | train.py | Unnecessary | **FIXED** |
| DUP-8.2 | — | loss.py / evaluation.py | Justified | Keep |
| DUP-8.3 | — | train.py / checkpoint.py / training_checkpoints.py | Justified | Keep |
| DUP-8.4 | — | train.py / lr.py | Justified | Keep |
| DUP-8.5 | — | utils.py / enhance.py | Justified | Keep |
| DUP-8.6 | — | train.py / convert.py | Justified | Keep |
| DUP-8.7 | — | train.py (MultiResolutionSTFTLoss) | Not duplication | N/A |
| DUP-8.8 | — | train.py (spectral_loss) / loss.py (SpectralLoss) | Justified | Keep |

## Implemented Consolidations

### DUP-8.1: Trainer.save_checkpoint → module-level save_checkpoint

- **Files:** `train.py`
- **Evidence:** `Trainer.save_checkpoint` (L893-934) reimplemented the identical
  atomic-write pattern (tree_flatten → mx.eval → mx.save_safetensors via tmp file →
  json.dump state via tmp file with fsync → atomic rename) that the standalone
  `save_checkpoint` function (L522-584) already provides.
- **Fix:** Delegated `Trainer.save_checkpoint` to the standalone function, passing
  `step`, `best_loss`, and `config` as keyword arguments (captured via `**extra_state`).
- **Semantic note:** The delegated version adds default `epoch=0` and `loss=0.0` keys
  to the state JSON. `Trainer.load_checkpoint` uses `.get()` with defaults, so extra
  keys are harmless.
- **LOC removed:** 29 → 23 in diff (net −6 lines of implementation logic, eliminating
  ~20 lines of duplicated atomic-write boilerplate).
- **Commit:** `418412b refactor(train): delegate Trainer.save_checkpoint to module-level save_checkpoint (DUP-8.1)`
- **Tests:** 977 passed, 11 skipped (unchanged from baseline)

## Justified Duplication (Keep)

### DUP-8.2: `si_sdr` — loss.py vs evaluation.py

- **loss.py:414** — returns `mean(si_sdr)` scalar for loss backprop, eps=1e-8
- **evaluation.py:27** — returns per-batch squeezed tensor for metric reporting, eps=1e-7
- **Rationale:** Different return semantics (scalar loss vs per-item metric), different
  eps values, different consumers. Merging would require conditional logic that hurts
  clarity.

### DUP-8.3: `save_checkpoint` / `load_checkpoint` — train.py vs checkpoint.py vs training_checkpoints.py

- **train.py** (L522/L588): Simple model+JSON checkpoint for standalone/example use.
  API: `save_checkpoint(model, path=..., step=..., **extra)`.
- **checkpoint.py** (L206/L289): Full training state with `CheckpointState`,
  optimizer/scheduler serialization, patience tracking, best-checkpoint management.
  API: `save_checkpoint(checkpoint_dir, model, state, optimizer, scheduler, is_best)`.
- **training_checkpoints.py** (L469/L631): Training-pipeline-specific with manifests,
  epoch markers, resume logic, discriminator weights, cleanup policies.
- **Rationale:** Three genuinely different API contracts for three different use cases
  (examples, standalone training, pipeline training). Consolidating would require a
  complex union API that serves none well.
- **Callers:** train.py versions used only by `examples/`; checkpoint.py by `train_gan.py`;
  training_checkpoints.py by `train_dynamic.py` and pipeline tests.

### DUP-8.4: `WarmupCosineSchedule` (train.py) vs `CosineScheduler` (lr.py)

- **WarmupCosineSchedule** (train.py:630): Random-access callable `__call__(step) → lr`.
  Supports arbitrary step lookup without sequential iteration. 4 params.
- **CosineScheduler** (lr.py:131): Sequential iterator `step() → lr`. Supports
  state_dict/load_state_dict for checkpointing, cycle decay, cycle multiplier. 8 params.
- **Rationale:** Different interface contracts. WarmupCosineSchedule provides O(1)
  random-access LR lookup by step number (used for checkpoint resume). CosineScheduler
  is a sequential iterator with richer features. Wrapping one around the other would
  either lose random access or require replaying all steps.
- **Callers:** WarmupCosineSchedule used by Trainer, train_dynamic, test_mlx,
  test_mlx_comprehensive, examples/. CosineScheduler used by checkpoint.py, train_gan.py.

### DUP-8.5: `load_audio` / `save_audio` — utils.py vs enhance.py

- **utils.py** (L22/L61): Simple load/save for `AudioDataset` and data pipeline use.
- **enhance.py** (L88/L124): Extended load/save with output directory management,
  suffix handling, and enhancement-pipeline integration.
- **Rationale:** Different feature sets for different pipelines. Neither is a strict
  subset of the other.

### DUP-8.6: `convert_pytorch_weights` / `load_pytorch_checkpoint` — train.py vs convert.py

- **train.py** (L976/L1015): Simple generic conversion for tutorials and examples.
- **convert.py** (L296/L375): Full conversion with model-type dispatch, transposition
  rules, DFNet3-specific handling.
- **Rationale:** train.py versions are simple entry points used by example scripts.
  convert.py is the canonical converter for production use. `__init__.py` exports both
  (train.py's as `load_pytorch_checkpoint`, convert.py's as `load_pytorch_ckpt`).

## Not Duplication

### DUP-8.7: MultiResolutionSTFTLoss (train.py) wrapping SpectralLoss (loss.py)

Already consolidated in Pass 7. `MultiResolutionSTFTLoss.__call__` delegates to
`SpectralLoss`. The wrapper adds `compute_per_resolution()` and `from_config()` — these
are value-add features, not duplication.

### DUP-8.8: `spectral_loss` function (train.py) vs `SpectralLoss` class (loss.py)

- **spectral_loss** (train.py:347): Operates on pre-computed STFT tuples
  `(real, imag)` — frequency-domain input.
- **SpectralLoss** (loss.py:100): Operates on waveforms and computes STFT internally —
  time-domain input.
- **Rationale:** Different input domains. `spectral_loss` is used in training loops
  where STFT is already computed as part of the model forward pass. `SpectralLoss`
  is used for waveform-level loss computation (MRSTFT).

## Residual Observations

1. **Re-export layers are intentional.** Both `__init__.py` and `train_dynamic.py`
   re-export from decomposed modules for backward compatibility. This is documented
   architecture, not duplication.
2. **training_waveform.py:compute_mrstft_loss** is a caller of `MultiResolutionSTFTLoss`,
   not a duplicate. It adds spec→wav conversion and FP32 stabilization.
3. **Pass 4–7 already removed significant dead code** (ASRLoss, create_loss_fn,
   MaskSpecLoss, compute_discriminator_loss, compute_generator_loss,
   create_discriminator, 4 dead train.py functions). The remaining code is either
   actively used or intentionally structured for different use cases.

## Metrics

| Metric | Value |
|--------|-------|
| Findings examined | 8 |
| Unnecessary duplication found | 1 |
| Consolidations implemented | 1 |
| LOC removed (net) | ~6 |
| Tests before | 977 passed, 11 skipped |
| Tests after | 977 passed, 11 skipped |
| Commits | 1 |
