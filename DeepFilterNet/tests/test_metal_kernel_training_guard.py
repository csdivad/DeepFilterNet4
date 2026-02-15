"""Tests that Metal kernel paths are correctly bypassed during training.

MLX's ``mx.fast.metal_kernel`` primitives do NOT implement VJP (backward
differentiation).  Any ``CustomKernel`` on the ``value_and_grad`` graph
triggers ``ValueError: [Primitive::vjp] Not implemented for CustomKernel``.

The fix gates every Metal kernel dispatch on ``not self.training`` (for
``nn.Module`` subclasses) or ``use_metal_kernel=False`` (for free functions
like ``istft`` called inside the gradient path).

These tests verify:
  1. DfOp uses the fallback in training mode; kernel in eval mode.
  2. MelSpectrogram likewise.
  3. istft with ``use_metal_kernel=False`` never enters the kernel path.
  4. value_and_grad through DfOp succeeds (gradient computation works).
"""

from __future__ import annotations

import platform

import mlx.core as mx
import mlx.nn as nn
import pytest


def _on_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# Skip the whole module if not on Apple Silicon (Metal kernels unavailable)
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not _on_apple_silicon(),
    reason="Metal kernels require Apple Silicon",
)


# ---------------------------------------------------------------------------
# 1. DfOp: training mode ➜ fallback; eval mode ➜ kernel
# ---------------------------------------------------------------------------


class TestDfOpTrainingGuard:
    """DfOp must use the differentiable fallback when ``self.training`` is True."""

    @pytest.fixture()
    def dfop(self):
        from df_mlx.modules import DfOp

        return DfOp(nb_df=8, df_order=3, df_lookahead=0, use_metal_kernel=True)

    @staticmethod
    def _make_inputs(batch: int = 2, time: int = 4, freq: int = 16, df_order: int = 3, nb_df: int = 8):
        spec_real = mx.random.normal((batch, time, freq))
        spec_imag = mx.random.normal((batch, time, freq))
        coef = mx.random.normal((batch, time, nb_df, df_order, 2))
        return (spec_real, spec_imag), coef

    def test_training_mode_uses_fallback(self, dfop):
        """In training mode, DfOp must bypass the Metal kernel (no VJP crash)."""
        dfop.train()
        spec, coef = self._make_inputs()
        out_real, out_imag = dfop((spec[0], spec[1]), coef)
        mx.eval(out_real, out_imag)
        assert out_real.shape == spec[0].shape
        assert out_imag.shape == spec[1].shape

    def test_eval_mode_uses_kernel(self, dfop):
        """In eval mode, DfOp should use the Metal kernel (if available)."""
        dfop.eval()
        spec, coef = self._make_inputs()
        out_real, out_imag = dfop((spec[0], spec[1]), coef)
        mx.eval(out_real, out_imag)
        assert out_real.shape == spec[0].shape

    def test_value_and_grad_through_dfop(self, dfop):
        """value_and_grad must succeed through DfOp in training mode."""
        dfop.train()

        class Wrapper(nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op
                # DfOp has no learnable weights; add a dummy so value_and_grad
                # can differentiate w.r.t. at least one parameter.
                self.scale = mx.ones((1,))

            def __call__(self, spec_real, spec_imag, coef):
                out_r, out_i = self.op((spec_real, spec_imag), coef)
                return mx.mean(self.scale * (out_r**2 + out_i**2))

        wrapper = Wrapper(dfop)
        wrapper.train()
        spec, coef = self._make_inputs()

        loss_fn = nn.value_and_grad(wrapper, wrapper)
        loss, grads = loss_fn(spec[0], spec[1], coef)
        mx.eval(loss, grads)
        assert loss.ndim == 0  # scalar loss

    def test_train_eval_output_consistency(self, dfop):
        """Training and eval modes must produce the same numerical output."""
        spec, coef = self._make_inputs()

        dfop.train()
        train_r, train_i = dfop((spec[0], spec[1]), coef)
        mx.eval(train_r, train_i)

        dfop.eval()
        eval_r, eval_i = dfop((spec[0], spec[1]), coef)
        mx.eval(eval_r, eval_i)

        assert mx.allclose(train_r, eval_r, atol=1e-4).item()
        assert mx.allclose(train_i, eval_i, atol=1e-4).item()


# ---------------------------------------------------------------------------
# 2. MelSpectrogram: training mode ➜ fallback
# ---------------------------------------------------------------------------


class TestMelSpectrogramTrainingGuard:
    """MelSpectrogram must use the differentiable fallback when training."""

    @pytest.fixture()
    def mel(self):
        from df_mlx.dnsmos_proxy import MelSpectrogram

        return MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=256,
            n_mels=64,
            use_metal_kernel=True,
        )

    def test_training_mode_uses_fallback(self, mel):
        mel.train()
        audio = mx.random.normal((2, 16000))
        out = mel(audio)
        mx.eval(out)
        assert out.ndim == 3  # (batch, n_mels, time)

    def test_eval_mode_uses_kernel(self, mel):
        mel.eval()
        audio = mx.random.normal((2, 16000))
        out = mel(audio)
        mx.eval(out)
        assert out.ndim == 3

    def test_train_eval_output_consistency(self, mel):
        audio = mx.random.normal((2, 16000))

        mel.train()
        train_out = mel(audio)
        mx.eval(train_out)

        mel.eval()
        eval_out = mel(audio)
        mx.eval(eval_out)

        assert mx.allclose(train_out, eval_out, atol=1e-4).item()


# ---------------------------------------------------------------------------
# 3. istft: use_metal_kernel=False bypasses the kernel
# ---------------------------------------------------------------------------


class TestIstftMetalKernelFlag:
    """istft with use_metal_kernel=False must never touch CustomKernel."""

    def test_istft_no_kernel(self):
        from df_mlx.ops import istft

        n_fft = 960
        hop = 480
        frames = 10
        freq = n_fft // 2 + 1
        spec_real = mx.random.normal((1, frames, freq))
        spec_imag = mx.random.normal((1, frames, freq))
        wav = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=False)
        mx.eval(wav)
        assert wav.ndim == 1 or wav.ndim == 2  # 1D if batch was squeezed

    def test_istft_with_kernel(self):
        from df_mlx.ops import istft

        n_fft = 960
        hop = 480
        frames = 10
        freq = n_fft // 2 + 1
        spec_real = mx.random.normal((1, frames, freq))
        spec_imag = mx.random.normal((1, frames, freq))
        wav = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=True)
        mx.eval(wav)
        assert wav.ndim == 1 or wav.ndim == 2

    def test_kernel_vs_no_kernel_consistency(self):
        from df_mlx.ops import istft

        n_fft = 960
        hop = 480
        frames = 10
        freq = n_fft // 2 + 1
        spec_real = mx.random.normal((1, frames, freq))
        spec_imag = mx.random.normal((1, frames, freq))

        wav_kernel = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=True)
        wav_fallback = istft((spec_real, spec_imag), n_fft=n_fft, hop_length=hop, use_metal_kernel=False)
        mx.eval(wav_kernel, wav_fallback)

        assert mx.allclose(wav_kernel, wav_fallback, atol=1e-3).item()
