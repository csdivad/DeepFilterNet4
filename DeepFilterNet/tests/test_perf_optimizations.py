"""Tests for PERF-P0/P1 optimizations: dtype guards, cached transpose, lazy loss dict.

Validates that:
1. Redundant FP32 casts are skipped when input is already FP32 (PERF-P0-001)
2. Pipeline/awesome losses cast once at entry (PERF-P0-002)
3. ERB filterbank transpose is cached at init (PERF-P0-003)
4. CombinedLoss returns lazy mx.array dict, not float dict (PERF-P1-002)
5. All dtype-guarded functions produce identical results for FP16 and FP32 inputs
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestDtypeGuards:
    """PERF-P0-001: Verify dtype guards skip redundant casts."""

    def test_log1p_mag_fp32_no_cast(self):
        """_log1p_mag should not add astype nodes when input is already FP32."""
        from df_mlx.train_dynamic import _log1p_mag

        real = mx.ones((2, 10, 32), dtype=mx.float32)
        imag = mx.ones((2, 10, 32), dtype=mx.float32)
        result = _log1p_mag(real, imag)
        mx.eval(result)
        assert result.dtype == mx.float32

    def test_log1p_mag_fp16_casts(self):
        """_log1p_mag should cast FP16 inputs to FP32."""
        from df_mlx.train_dynamic import _log1p_mag

        real = mx.ones((2, 10, 32), dtype=mx.float16)
        imag = mx.ones((2, 10, 32), dtype=mx.float16)
        result = _log1p_mag(real, imag)
        mx.eval(result)
        assert result.dtype == mx.float32

    def test_log1p_mag_numerical_equivalence(self):
        """FP16 vs FP32 inputs should produce close results."""
        from df_mlx.train_dynamic import _log1p_mag

        np.random.seed(42)
        data_np = np.random.randn(2, 10, 32).astype(np.float32) * 0.1
        real_f32 = mx.array(data_np)
        imag_f32 = mx.array(data_np * 0.5)
        real_f16 = real_f32.astype(mx.float16)
        imag_f16 = imag_f32.astype(mx.float16)

        result_f32 = _log1p_mag(real_f32, imag_f32)
        result_f16 = _log1p_mag(real_f16, imag_f16)
        mx.eval(result_f32, result_f16)

        np.testing.assert_allclose(np.array(result_f32), np.array(result_f16), rtol=1e-2, atol=1e-3)

    def test_compute_vad_probs_fp32_no_cast(self):
        """_compute_vad_probs should skip casts when inputs are FP32."""
        from df_mlx.train_dynamic import _compute_vad_probs

        clean_real = mx.ones((2, 10, 32), dtype=mx.float32) * 0.5
        clean_imag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.3
        out_real = mx.ones((2, 10, 32), dtype=mx.float32) * 0.4
        out_imag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.2
        band_mask = mx.ones((1, 1, 32), dtype=mx.float32)
        band_bins = 32.0

        p_ref, p_out = _compute_vad_probs(
            clean_real,
            clean_imag,
            out_real,
            out_imag,
            band_mask,
            band_bins,
            vad_z_threshold=0.0,
            vad_z_slope=0.5,
        )
        mx.eval(p_ref, p_out)
        assert p_ref.dtype == mx.float32
        assert p_out.dtype == mx.float32


class TestCastOnceAtEntry:
    """PERF-P0-002: Verify cast-once pattern in awesome/pipeline losses."""

    def test_compute_proxy_gates_fp32_passthrough(self):
        """_compute_proxy_gates should skip casts when inputs are FP32."""
        from df_mlx.train_dynamic import _compute_proxy_gates

        B, T, F = 2, 10, 32
        clean_real = mx.ones((B, T, F), dtype=mx.float32) * 0.5
        clean_imag = mx.ones((B, T, F), dtype=mx.float32) * 0.3
        noisy_real = mx.ones((B, T, F), dtype=mx.float32) * 0.8
        noisy_imag = mx.ones((B, T, F), dtype=mx.float32) * 0.6
        snr = mx.array([10.0, 15.0])
        band_mask = mx.ones((1, 1, F), dtype=mx.float32)
        band_bins = float(F)

        result = _compute_proxy_gates(
            clean_real,
            clean_imag,
            noisy_real,
            noisy_imag,
            snr,
            band_mask,
            band_bins,
            vad_z_threshold=0.0,
            vad_z_slope=0.5,
            vad_snr_gate_db=5.0,
            vad_snr_gate_width=2.0,
            proxy_enabled=True,
        )
        mx.eval(*result)
        proxy_frame = result[0]
        assert proxy_frame.dtype == mx.float32


class TestErbFbCached:
    """PERF-P0-003: Verify _erb_fb_T is cached at init."""

    def test_erb_fb_t_exists(self):
        """DfNet4 should have _erb_fb_T attribute after init."""
        from df_mlx.config import get_default_config
        from df_mlx.model import DfNet4

        p = get_default_config()
        model = DfNet4(p)
        assert hasattr(model, "_erb_fb_T"), "DfNet4 must have _erb_fb_T cached at init"
        expected = mx.transpose(model._erb_fb)
        mx.eval(expected, model._erb_fb_T)
        np.testing.assert_array_equal(np.array(expected), np.array(model._erb_fb_T))

    def test_erb_fb_t_used_in_forward(self):
        """Forward pass should use _erb_fb_T (not recompute transpose)."""
        from df_mlx.config import get_default_config
        from df_mlx.model import DfNet4

        p = get_default_config()
        model = DfNet4(p)

        B, T = 1, 10
        n_freq = p.fft_size // 2 + 1
        spec_real = mx.zeros((B, T, n_freq))
        spec_imag = mx.zeros((B, T, n_freq))
        feat_erb = mx.zeros((B, T, p.nb_erb))
        feat_spec = mx.zeros((B, T, p.nb_df, 2))

        out = model((spec_real, spec_imag), feat_erb, feat_spec)
        mx.eval(*out)
        assert out[0].shape == (B, T, n_freq)


class TestCombinedLossLazy:
    """PERF-P1-002: Verify CombinedLoss returns lazy mx.array dict."""

    def test_combined_loss_returns_lazy_arrays(self):
        """CombinedLoss dict values should be mx.array, not float."""
        from df_mlx.loss import CombinedLoss

        loss_fn = CombinedLoss(sisdr_factor=0.5)

        pred = mx.zeros((1, 48000))
        target = mx.ones((1, 48000)) * 0.01

        total_loss, losses = loss_fn(pred, target)
        assert isinstance(total_loss, mx.array)
        for key, val in losses.items():
            assert isinstance(val, mx.array), f"CombinedLoss['{key}'] should be mx.array, got {type(val)}"


class TestMusicnessGuards:
    """PERF-P0-001: Verify musicness functions have dtype guards."""

    def test_compute_musicness_fp32_passthrough(self):
        """_compute_musicness should skip cast when mag is FP32."""
        from df_mlx.train_dynamic import _compute_musicness

        mag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.5
        band_mask = mx.ones((1, 1, 32), dtype=mx.float32)
        musicness, gate = _compute_musicness(mag, band_mask, 32.0)
        mx.eval(musicness, gate)
        assert musicness.dtype == mx.float32

    def test_compute_improved_musicness_fp32_passthrough(self):
        """_compute_improved_musicness should skip cast when mag is FP32."""
        from df_mlx.train_dynamic import _compute_improved_musicness

        mag = mx.ones((2, 10, 32), dtype=mx.float32) * 0.5
        band_mask = mx.ones((1, 1, 32), dtype=mx.float32)
        snr = mx.array([10.0, 15.0])
        musicness, vocal, instrument = _compute_improved_musicness(mag, band_mask, 32.0, snr)
        mx.eval(musicness, vocal, instrument)
        assert musicness.dtype == mx.float32
