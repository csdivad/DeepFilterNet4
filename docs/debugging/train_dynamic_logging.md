# Training Loop Logging — Definitions & Debugging

## Counter Definitions

| Counter | Semantics | Increments When |
|---------|-----------|-----------------|
| `epoch` | 1-indexed epoch number | Start of each epoch loop iteration |
| `batch_idx` | 0-indexed batch within current epoch | Each batch from data iterator |
| `global_step` | Monotonically increasing optimizer step | Optimizer updates (after grad accumulation completes) |
| `micro_step` | Implicit via `micro_batches_in_accum` | Each forward pass within a grad accumulation window |

**With gradient accumulation** (`--grad-accumulation-steps N`):
- `batch_idx` increments every micro-batch
- `global_step` increments every N micro-batches (when accumulation window completes)
- Effective batch size = `batch_size × grad_accumulation_steps`

## Progress Bar Fields

### Verbose mode (`--verbose`)

| Field | Meaning |
|-------|---------|
| `loss` | Current batch loss |
| `spec` | Spectral loss component |
| `fwd` | Forward+backward+update time in ms |
| `spd` | Throughput (samples/second), averaged over sync window |
| `gstep` | Global optimizer step counter |
| `data` | Data loading time in ms |
| `lr` | Current learning rate |

### Standard mode

| Field | Meaning |
|-------|---------|
| `loss` | Current batch loss |
| `avg` | Running average loss |
| `spd` | Throughput (samples/second) |
| `gstep` | Global optimizer step counter |
| `grad` | Gradient norm |
| `lr` | Current learning rate |

## Throughput Measurement

Throughput (`spd`) is computed as:

```
samples_per_sec = window_samples / window_elapsed
```

Where:
- `window_samples`: total samples processed since last sync point
- `window_elapsed`: wall-clock time since last sync point (using `time.perf_counter()`)

The sync window is determined by `--eval-frequency` (default 10). Every `eval_frequency` batches, MLX tensors are synchronized and metrics are logged.

**Why `time.perf_counter()`**: Unlike `time.time()`, `perf_counter()` uses a monotonic clock that is immune to NTP adjustments and system clock changes.

## Resuming Training

On resume from checkpoint:
1. `global_step` is restored from checkpoint metadata
2. Epoch index is restored from `last_completed_epoch + 1`
3. Scheduler state is reconstructed from `global_step`
4. A banner is printed showing restored values

```bash
# Resume from latest checkpoint
python -m df_mlx.train_dynamic \
    --config dataset_config.json \
    --resume-from checkpoints/best.safetensors \
    --epochs 100
```

## Known Gotchas

### Output Corruption (tqdm + print)
All messages inside the batch training loop use `tqdm.write()` instead of `print()`.
This prevents carriage-return collisions between tqdm's progress bar and log messages.

**Environment variable**: Set `DFNET_TQDM=off` to disable progress bars entirely
(useful when logging to files).

### Buffering
Use `PYTHONUNBUFFERED=1` when capturing logs to avoid delayed output:

```bash
PYTHONUNBUFFERED=1 python -m df_mlx.train_dynamic ... 2>&1 | tee train.log
```

### Multiple Progress Writers
tqdm is configured to write to stderr. Redirect stdout for clean log capture:

```bash
python -m df_mlx.train_dynamic ... > train_stdout.log 2> train_stderr.log
```

## Verification Commands

```bash
# Run logging integrity tests
cd DeepFilterNet && python -m pytest tests/test_train_logging_integrity.py -v

# Dry-run training to verify output format (requires data)
PYTHONUNBUFFERED=1 python -m df_mlx.train_dynamic \
    --config dataset_config.json \
    --max-train-batches 20 \
    --epochs 1 \
    --verbose 2>&1 | tee /tmp/train_check.log

# Verify log lines are not corrupted (each line should be complete)
grep -c $'\r' /tmp/train_check.log  # should be 0 for redirected output
```
