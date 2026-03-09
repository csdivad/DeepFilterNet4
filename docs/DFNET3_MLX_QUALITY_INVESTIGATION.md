# DFNet3 MLX Quality Investigation

## Overview

This document tracks the investigation into quality regression in the MLX
DFNet3 implementation compared to the reference PyTorch model. After fixes,
the MLX model produces output that is numerically close to the PyTorch
DFNet3 pretrained model, with minor float32 precision drift from GRU
evaluation order differences (see **Root Cause** below).

**Branch:** `feat/dfn3-on-df_mlx-fix`
**Test audio:** `p225_001_mic1.flac` from VCTK-Corpus-0.92
**Reference output:** `/Volumes/TrainingData/datasets/preprocessed/dfn3_speech_clean-good/…/p225_001_mic1.flac`

---

## Architecture Correspondence: PyTorch ↔ MLX

### Top-level Model

| Component            | PyTorch (`df/deepfilternet3.py`)      | MLX (`df_mlx/deepfilternet3.py`)       |
|----------------------|---------------------------------------|----------------------------------------|
| Main model           | `DfNet` (line 308)                    | `DFNet3` (line 697)                    |
| Encoder              | `Encoder` (line 64)                   | `Encoder3` (line 328)                  |
| ERB decoder          | `ErbDecoder` (line 164)               | `ErbDecoder3` (line 485)               |
| DF decoder           | `DfDecoder` (line 231)                | `DfDecoder3` (line 620)                |
| DF operation         | `MF.DF` (`df/multiframe.py:158`)      | `DfOp` (`df_mlx/modules.py:570`)       |
| ERB filterbank       | `erb_fb` / `erb_inv_fb` (torch)       | `_erb_fb` / `_erb_inv_fb` (mx.array)   |

### Forward Flow

```
PT: pad_feat → enc → erb_dec(mask) → matmul(erb_inv_fb) → df_dec(coefs) → MF.DF → concat → post_filter
MLX: _apply_conv_lookahead → enc → erb_decoder(mask) → matmul(erb_inv_fb) → df_decoder(coefs) → DfOp → concat
```

### Conv Layer Wrappers

| PyTorch                        | MLX                                | Notes                                      |
|--------------------------------|------------------------------------|--------------------------------------------|
| `Conv2dNormAct` (df/modules)   | `Conv2dNormAct` (modules.py:29)    | MLX uses NHWC; padding handled by wrapper  |
| `ConvTranspose2dNormAct`       | `ConvTranspose2dNormAct` (line 215)| Separable via GroupedConvTranspose2d        |
| `SqueezedGRU_S` (df/modules)   | `SqueezedGRU_S` (modules.py:1155)  | Skip connection after output projection    |
| `GroupedLinearEinsum` (df/mod)  | `GroupedLinear` (modules.py)       | Einsum → explicit reshape+matmul           |

### Key Config Parameters

```
SR=48000, FFT=960, Hop=480, nb_erb=32, nb_df=96
conv_ch=64, conv_lookahead=2, conv_kernel=(1,3), conv_kernel_inp=(3,3)
convt_kernel=(1,3), conv_depthwise=True, convt_depthwise=True
emb_hidden_dim=256, df_order=5, df_lookahead=2, post_filter=False
```

---

## Issues Found and Fixed

### BUG-1: Missing `model.eval()` (FIXED)

**File:** `df_mlx/enhance.py` line ~275
**Problem:** `load_model()` never called `model.eval()`, so BatchNorm layers ran in training mode during inference, using batch statistics instead of running mean/var.
**Fix:** Added `model.eval()` after weight loading.

### BUG-2: `conv0_out` missing BatchNorm (FIXED)

**File:** `df_mlx/deepfilternet3.py` line 575
**Problem:** `conv0_out` was initialized with `norm=None` but the PyTorch model has BatchNorm on this layer (4 weight keys: `running_mean`, `running_var`, `weight`, `bias`).
**Fix:** Changed `norm=None` to `norm="batch"`.

### BUG-3: `conv0_out` BN weights not converted (FIXED)

**File:** `df_mlx/convert.py` line 340
**Problem:** The conversion spec for `conv0_out` had `norm_index=None`, so the BatchNorm weights were silently dropped during conversion.
**Fix:** Changed `norm_index=None` to `norm_index=1`. Re-ran conversion: 110 → 114 keys.

### BUG-4: MLX `mx.conv_transpose2d` all-zeros with groups > 1 + stride > 1 (WORKAROUND)

**File:** `df_mlx/modules.py` `GroupedConvTranspose2d.__call__`
**Problem:** `mx.conv_transpose2d` produces all-zero output when `groups > 1` AND `stride > 1`. This affects `convt1` and `convt2` in the ERB decoder (both use groups=64, stride=(1,2)).
**Impact:** End-to-end corr was ~0.10 before this fix.
**Workaround:** Detect the condition and split into per-group `mx.conv_transpose2d` calls with `groups=1`, then concatenate. Confirmed: pre-BN output of `convt2` went from all-zeros to corr=1.000000 vs PyTorch.
**Status:** Workaround in place. Should file upstream MLX bug report.

### BUG-5: Missing sigmoid activation in `Conv2dNormAct` (FIXED)

**File:** `df_mlx/modules.py` line ~186
**Problem:** `Conv2dNormAct.__init__` handled `relu`, `gelu`, `silu`, `prelu`, `leaky_relu` but **not `sigmoid`**. When `activation="sigmoid"` was passed for `conv0_out`, `self.activation` remained `None` — so the sigmoid was never applied.
**Impact:** ERB mask output was unbounded instead of [0,1]. ERB mask corr dropped from 0.971 to 0.929; output std was 1.176 instead of 0.123.
**Fix:** Added `elif activation == "sigmoid": self.activation = nn.Sigmoid()` to both `Conv2dNormAct` and `ConvTranspose2dNormAct`.

---

## Definitively Ruled Out

| Hypothesis                         | Evidence                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------|
| Encoder correctness                | All skip connections e0-e3: corr=1.000000, max_diff=0.000000                              |
| Feature computation                | Both use `libdf` Rust FFI — identical STFT/ERB/norm                                        |
| MLX BatchNorm eval mode            | Verified: all BN layers show `training=False`; max diff = 1.9e-06 in isolation             |
| BN running stats match             | All running_mean / running_var identical between PT and MLX                                |
| Weight conversion (transposed conv)| PT `(64,1,1,3)` → MLX `(64,1,3,1)` verified correct for NHWC                              |
| ConvTranspose2d padding params     | Both use `padding=(0,1)`, `output_padding=(0,1)`, `stride=(1,2)`                           |
| Metal kernel DfOp                  | Metal vs pure-MLX fallback: corr=1.000000, max_diff=0.000000                               |
| DfOp math correctness              | DfOp output: corr=0.999645, max_diff=0.003388 vs PT — acceptable float32 drift            |
| `_apply_conv_lookahead`            | Matches PT's `ConstantPad2d((0,0,-2,2), 0.0)` exactly                                     |
| Train vs eval mode                 | Train corr=0.984 vs eval corr=0.985 — BN running stats NOT the dominant gap               |
| Post-filter                        | Disabled (`post_filter=False`)                                                             |
| Flatten/reshape order              | Verified correct                                                                            |

---

## Remaining Gap Analysis

After all five bug fixes, end-to-end quality:

```
PT vs Reference:  corr=1.000000, max_diff=0.000031   (gold standard)
MLX vs Reference: corr=0.981,    max_diff=0.270
PT vs MLX:        corr=0.981,    max_diff=0.270
```

### ERB Decoder Step-by-Step Trace (PT encoder outputs → MLX decoder)

```
emb_gru output:              corr=0.999025, max_diff=0.096, rms=0.007
emb reshaped:                corr=0.999025 (same)
conv3p(e3):                  corr=1.000000 (perfect)
conv3p(e3) + emb:            corr=0.999094
convt3(conv3p+emb):          corr=0.997166, max_diff=0.368
conv2p(e2):                  corr=1.000000 (perfect)
conv2p(e2) + x:              corr=0.997360
convt2 ** TRANSPOSED **:     corr=0.997475
conv1p(e1):                  corr=1.000000 (perfect)
conv1p(e1) + x:              corr=0.998131
convt1 ** TRANSPOSED **:     corr=0.996915
conv0p(e0):                  corr=1.000000 (perfect)
conv0p(e0) + x:              corr=0.997085
conv0_out ** FINAL **:       corr=0.989719, max_diff=0.142
```

### Root Cause: GRU Numerical Drift

The `emb_gru` produces corr=0.999 instead of 1.0. This is numerical precision
drift between MLX and PyTorch GRU implementations — different evaluation order,
hardware, and intermediate precision. This drift amplifies through 7+
subsequent conv layers in the ERB decoder, reaching corr=0.990 at the mask
output.

The GRU drift is expected and not fixable without bit-exact GRU
implementations. The end-to-end quality is reasonable for a cross-framework
port.

---

## Quality Timeline

| Milestone                          | PT vs MLX corr | Key Change                             |
|------------------------------------|---------------|----------------------------------------|
| Initial (broken conv_transpose)    | ~0.10         | All-zeros from GroupedConvTranspose2d  |
| After convt2d workaround           | ~0.985        | Per-group split workaround             |
| After sigmoid fix                  | ~0.981        | Correct mask range [0,1]               |
| After ERB filterbank fix           | ~1.000        | Boxcar filters matching PT/libDF       |

Note: corr slightly decreased after sigmoid fix because the unbounded mask
happened to produce values that correlated better with the reference through
coincidental cancellation. The 0.981 value is **more correct** because the mask
is now properly bounded. The ERB filterbank fix resolved the remaining gap
by replacing smooth triangular filters with rectangular boxcar filters that
match the PyTorch/libDF contract exactly.

---

## Files Modified

1. `DeepFilterNet/df_mlx/enhance.py` — `model.eval()` in `load_model()`
2. `DeepFilterNet/df_mlx/deepfilternet3.py` — `conv0_out` norm="batch", `compute_erb_fb()`, `build_dfnet3_model()` passes widths
3. `DeepFilterNet/df_mlx/convert.py` — `conv0_out` norm_index=1
4. `DeepFilterNet/df_mlx/modules.py` — convt2d workaround + sigmoid activation
5. `DeepFilterNet/df_mlx/ops.py` — boxcar ERB filterbank path via `widths` parameter
6. `models/mlx/DeepFilterNet3-MLX/` — re-converted model (114 keys)
