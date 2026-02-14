# Repository Standards & Conventions

## 1. Scope and Purpose

This file captures non-obvious, repo-specific rules that matter for correctness, maintainability, and team sanity. These are patterns that a new contributor would benefit from knowing explicitly.

For general coding standards (formatting, linting, commit messages), see [CONTRIBUTING.md](../CONTRIBUTING.md).

## 2. Core Conventions

### Model Version Architecture Pattern

**Status:** REQUIRED

**Scope:** All `DeepFilterNet/df/deepfilternet*.py` files

**Rule:**

- Each model version (DFNet, DFNet2, DFNet3, DFNet4) has its own module file following the naming pattern `deepfilternet{N}.py`.
- New model versions extend the architecture via composition, not inheritance from previous versions.
- Model-specific configuration uses the `DfnetConfig` section in config files.

**Rationale:**

- Keeps model architectures isolated and independently testable.
- Allows comparing performance across versions without coupling.
- Prevents regression in older models when experimenting with new approaches.

**Related Files:**

- [DeepFilterNet/df/deepfilternet.py](../DeepFilterNet/df/deepfilternet.py)
- [DeepFilterNet/df/deepfilternet4.py](../DeepFilterNet/df/deepfilternet4.py)

---

### Configuration Hierarchy

**Status:** REQUIRED

**Scope:** All training, evaluation, and inference code

**Rule:**

- Model checkpoints expect a directory containing `config.ini` (or `config.yaml`) plus a `checkpoints/` subdirectory.
- Configuration is loaded via `df.config.DfParams` using INI or YAML parsers.
- Command-line arguments override config file values.

**Rationale:**

- Ensures reproducibility: models are always paired with their training config.
- Standard directory layout allows scripts to auto-discover model parameters.

**Examples:**

- Good: `model_dir/config.ini` + `model_dir/checkpoints/model_0001.pth`
- Bad: Loose `.pth` files without accompanying config

**Related Files:**

- [DeepFilterNet/df/config.py](../DeepFilterNet/df/config.py)
- [DeepFilterNet/df/checkpoint.py](../DeepFilterNet/df/checkpoint.py)

---

### Dual Language Crate Pattern

**Status:** REQUIRED

**Scope:** All Rust crates with Python bindings

**Rule:**

- `libDF/` contains pure Rust DSP and runtime code.
- `pyDF/` wraps `libDF` as Python bindings via PyO3/Maturin.
- `pyDF-data/` provides Rust-backed data loading for training.
- Never put Python-specific logic in `libDF/`.

**Rationale:**

- Keeps the Rust core portable (WebAssembly, C FFI, standalone CLI).
- Python bindings are a separate concern that shouldn't pollute core algorithms.

**Related Files:**

- [libDF/](../libDF/)
- [pyDF/](../pyDF/)
- [pyDF-data/](../pyDF-data/)

---

### ERB (Equivalent Rectangular Bandwidth) Scale

**Status:** REQUIRED

**Scope:** All spectral feature code

**Rule:**

- Spectral features use ERB-scale compression (default 32 bands) for the encoder.
- DF (Deep Filtering) operates on linear-frequency bins (default 96 lowest bins).
- ERB band count and DF bin count are configurable but defaults should be used unless experimenting.

**Rationale:**

- ERB scale matches human auditory perception, improving model efficiency.
- Linear DF bins focus compute on perceptually important low frequencies.

**Related Files:**

- [DeepFilterNet/df/modules.py](../DeepFilterNet/df/modules.py)
- [docs/ARCHITECTURE.md](ARCHITECTURE.md)

---

### Test Markers for Hardware-Specific Tests

**Status:** REQUIRED

**Scope:** All pytest tests

**Rule:**

- Use `@pytest.mark.mps` for Apple Silicon (MPS) specific tests.
- Tests that require GPU should be skippable via markers or environment checks.

**Rationale:**

- CI runs on Linux CPU instances; MPS tests would fail.
- Allows developers on Apple Silicon to run the full test suite with appropriate filtering.

**Examples:**

- Good: `@pytest.mark.mps` on tests that use `torch.device("mps")`
- Bad: Unconditionally creating MPS tensors in shared test fixtures

**Related Files:**

- [DeepFilterNet/tests/](../DeepFilterNet/tests/)
- [DeepFilterNet/pyproject.toml](../DeepFilterNet/pyproject.toml)

---

### Issue Tracking with bd (beads)

**Status:** REQUIRED

**Scope:** All AI agents and contributors

**Rule:**

- Use `bd` for issue tracking, not GitHub Issues directly.
- Run `bd prime` at session start for workflow context.
- Run `bd sync` before ending a work session.
- Reference issue IDs in commit messages when applicable.

**Rationale:**

- Git-backed issues stay with the repository and work offline.
- AI agents get structured context injection via bd.
- See `.claude/skills/beads/` for comprehensive AI integration patterns.

**Related Files:**

- [.beads/](../.beads/)
- [.claude/skills/beads/](../.claude/skills/beads/)
- [AGENTS.md](../AGENTS.md)

---

### Training Counter Semantics (MLX Dynamic Trainer)

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py`, `DeepFilterNet/df_mlx/dynamic_dataset.py`, and related checkpoint/resume tests

**Rule:**

- Treat epoch progress in **micro-batches** (dataloader iterations), and treat `global_step` in **optimizer steps** (parameter updates).
- In checkpoint state, `batch_idx` represents **micro-batches completed in the current epoch** (count), not the 0-based index of the last seen batch.
- Progress bars must use micro-batch totals and iterate over a bounded iterator to avoid overshooting epoch boundaries.
- When both model and data checkpoints are present for in-progress resume, their `(epoch, micro-batch)` positions must match exactly; otherwise fail loudly.

**Rationale:**

- Mixing units (optimizer-step totals vs micro-batch iteration) causes misleading progress bars and apparent epoch overruns.
- Count-vs-index ambiguity in `batch_idx` causes off-by-one resume behavior and can duplicate or skip data after interruption.
- Strict model/data checkpoint reconciliation prevents silent divergence during recovery and makes incidents reproducible.

**Related Files:**

- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)
- [DeepFilterNet/df_mlx/dynamic_dataset.py](../DeepFilterNet/df_mlx/dynamic_dataset.py)
- [DeepFilterNet/tests/test_train_control_semantics.py](../DeepFilterNet/tests/test_train_control_semantics.py)

---

### Epoch-Boundary Training Mode Switch (Compiled → Eager with GAN)

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` and trainer control-flow tests

**Rule:**

- Determine training mode once per epoch (before the batch loop), never mid-epoch.
- Allow compiled mode only while GAN is inactive and compiled-step base constraints are satisfied.
- Once GAN-active epochs begin, switch to eager mode and do not switch back to compiled mode later in the run.
- Emit explicit mode markers (`TRAIN_MODE=COMPILED` / `TRAIN_MODE=EAGER`) and fail fast if a GAN-active epoch tries to run the compiled step.

**Rationale:**

- Prevents mixed execution semantics within a single epoch and keeps checkpoint/resume behavior deterministic.
- Preserves pre-GAN performance gains from compiled training while ensuring GAN/discriminator updates run on the eager path.
- One-way switching avoids subtle resume-dependent mode oscillation and reduces incident surface area.

**Related Files:**

- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)
- [DeepFilterNet/tests/test_train_control_semantics.py](../DeepFilterNet/tests/test_train_control_semantics.py)

---

### Sync Mode Selection

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` and run-config TOML files

**Rule:**

- Every training run must specify a `sync_mode` (`fast`, `normal`, `debug`, or `profile`) that controls eval barrier frequency and metric verbosity.
- When `sync_mode` is set and `eval_frequency` is at its default, `eval_frequency` is automatically overridden to the mode's recommended value (`fast`→50, `normal`→10, `debug`→1, `profile`→5).
- An explicit `eval_frequency` in the TOML or CLI always takes precedence over the mode default.

**Rationale:**

- Prevents accidental performance loss from too-frequent synchronization in production runs.
- Provides a single knob that coordinates eval frequency, logging verbosity, and sync budget so developers don't need to tune multiple flags independently.

**Examples:**

- Set `sync_mode = "fast"` for throughput-critical production runs (eval every 50 steps).
- Set `sync_mode = "debug"` when investigating training anomalies (eval every step, full observability).

```toml
[debug]
sync_mode = "fast"
```

**Related Files:**

- [docs/SYNC_BARRIER_POLICY.md](SYNC_BARRIER_POLICY.md)
- [docs/RUN_CONFIG_PRESETS.md](RUN_CONFIG_PRESETS.md)

---

### Compile Boundary Shape Invariants

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` — all `mx.compile()`-wrapped functions

**Rule:**

- All inputs to `mx.compile()`-wrapped functions must have fixed shapes within a run.
- `drop_last=True` is mandatory on all data loaders feeding compiled paths to prevent tail batches with different shapes.
- Shape changes trigger expensive retraces that silently degrade throughput.

**Rationale:**

- MLX's `mx.compile()` traces the computation graph for a given set of input shapes. A shape change forces a full retrace, which is as expensive as the first compilation and silently destroys any throughput gains.
- Enforcing `drop_last=True` eliminates the most common source of shape variation (short tail batches at epoch boundaries).

**Examples:**

- Good: `PrefetchDataLoader(dataset, batch_size=4, drop_last=True)`
- Bad: `PrefetchDataLoader(dataset, batch_size=4, drop_last=False)` — the last batch may have fewer samples, triggering a retrace.

**Related Files:**

- [docs/COMPILE_BOUNDARY_AUDIT.md](COMPILE_BOUNDARY_AUDIT.md)
- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)

---

### Hardware Profile Presets

**Status:** STRONGLY RECOMMENDED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` training configuration

**Rule:**

- Use `--preset` with the appropriate hardware class (`entry`, `pro`, `max`, `ultra`, `debug`) instead of manually tuning worker/prefetch/batch parameters.
- Presets encode tested, hardware-specific defaults that balance throughput and memory.
- Preset values can still be overridden by explicit CLI flags or run-config TOML entries.

**Rationale:**

- Manual tuning of `num_workers`, `prefetch_size`, `batch_size`, and `sync_mode` is error-prone and requires hardware-specific knowledge.
- Presets were validated against the benchmark contract matrix and represent known-good configurations for each Apple Silicon tier.

**Examples:**

```bash
# For M3 Max hardware
python -m df_mlx.train_dynamic --preset max --run-config my_config.toml

# For debugging on any hardware
python -m df_mlx.train_dynamic --preset debug --run-config my_config.toml
```

**Related Files:**

- [docs/RUN_CONFIG_PRESETS.md](RUN_CONFIG_PRESETS.md)
- [docs/DATA_PIPELINE_TUNING.md](DATA_PIPELINE_TUNING.md)

---

### Performance Regression Gate

**Status:** REQUIRED

**Scope:** Changes to training loop, data pipeline, model code in `DeepFilterNet/df_mlx/`

**Rule:**

- Any performance-sensitive change must pass the benchmark regression gate before merge.
- The gate compares candidate benchmark results against a known baseline using the thresholds defined in the benchmark contract.
- Regressions exceeding the threshold require explicit justification or mitigation before merge.

**Rationale:**

- Prevents silent throughput degradation that accumulates across multiple changes.
- The automated gate (`scripts/perf_gate.py`) provides objective pass/fail signals, removing subjective judgment from performance review.

**Examples:**

```bash
# Generate baseline, then candidate, then compare
python -m df_mlx.benchmark_train_step --contract --metadata --json-out baseline.jsonl
python -m df_mlx.benchmark_train_step --contract --metadata --json-out candidate.jsonl
python scripts/perf_gate.py --baseline baseline.jsonl --candidate candidate.jsonl
```

**Related Files:**

- [docs/PERF_REGRESSION_GATE.md](PERF_REGRESSION_GATE.md)
- [docs/BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md)
- [scripts/perf_gate.py](../scripts/perf_gate.py)

---

### GAN-Phase Eager Mode

**Status:** REQUIRED

**Scope:** `DeepFilterNet/df_mlx/train_dynamic.py` — GAN training phase

**Rule:**

- GAN-active training must use eager mode unless `gan.experimental_compile=true` is explicitly set in the run-config TOML.
- The compiled→eager mode switch at GAN activation is one-way and irreversible within a run.
- The `gan.experimental_compile` flag gates an R&D experiment path and is not recommended for production training.

**Rationale:**

- Discriminator updates introduce additional sync barriers (`mx.eval()` per discriminator step) that are incompatible with the current compile boundary.
- One-way switching prevents resume-dependent mode oscillation and keeps checkpoint semantics deterministic.
- The experimental compile flag provides a controlled path for future R&D without relaxing the default safety constraint.

**Examples:**

```toml
# Default (recommended): GAN phase uses eager mode
[gan]
start_epoch = 10
# experimental_compile defaults to false

# R&D only: attempt compiled GAN phase
[gan]
start_epoch = 10
experimental_compile = true
```

**Related Files:**

- [docs/GAN_COMPILE_EXPERIMENT.md](GAN_COMPILE_EXPERIMENT.md)
- [docs/COMPILE_BOUNDARY_AUDIT.md](COMPILE_BOUNDARY_AUDIT.md)
- [DeepFilterNet/df_mlx/train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py)

---

## 3. Known Exceptions

_None documented yet._

## 4. Change History (Human-Readable)

- **2025-01-06**: Initial conventions document created during AI integration optimization.
- **2026-02-13**: Added MLX training counter semantics convention (micro-batch vs optimizer-step units, checkpoint resume invariants).
- **2026-02-13**: Added epoch-boundary compiled→eager training mode convention for delayed GAN activation.
- **2026-02-14**: Added sync mode selection, compile boundary shape invariants, hardware profile presets, performance regression gate, and GAN-phase eager mode conventions.
