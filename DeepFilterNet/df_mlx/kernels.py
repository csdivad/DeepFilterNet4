"""Custom Metal kernels for DfOp and other hotspots.

This module provides fused GPU kernels via ``mx.fast.metal_kernel`` that
replace multi-dispatch pure-MLX paths with single-dispatch Metal shader
programs.  A runtime availability check (``metal_kernels_available``)
lets callers fall back to pure-MLX when the API is absent.
"""

from typing import Tuple

import mlx.core as mx

_METAL_AVAILABLE: bool = hasattr(mx.fast, "metal_kernel")


def metal_kernels_available() -> bool:
    """Return True when ``mx.fast.metal_kernel`` is usable."""
    return _METAL_AVAILABLE


# ---------------------------------------------------------------------------
# DfOp: fused gather + complex MAC
# ---------------------------------------------------------------------------

_DFOP_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;

    // Decode flat index -> (b, t, f)
    int nb_df      = coef_real_shape[2];
    int time_steps = coef_real_shape[1];
    int df_order   = coef_real_shape[3];

    int f = elem % nb_df;
    int t = (elem / nb_df) % time_steps;
    int b = elem / (nb_df * time_steps);

    // Padded-time length lives in the spec shape array.
    int spec_padded_time = spec_real_pad_shape[1];

    // Accumulate complex MAC over taps
    T acc_real = 0;
    T acc_imag = 0;

    for (int k = 0; k < df_order; k++) {
        int spec_idx = b * spec_padded_time * nb_df
                     + (t + k) * nb_df
                     + f;
        int coef_idx = b * time_steps * nb_df * df_order
                     + t * nb_df * df_order
                     + f * df_order
                     + k;

        T sr = spec_real_pad[spec_idx];
        T si = spec_imag_pad[spec_idx];
        T cr = coef_real[coef_idx];
        T ci = coef_imag[coef_idx];

        acc_real += cr * sr - ci * si;
        acc_imag += cr * si + ci * sr;
    }

    out_real[elem] = acc_real;
    out_imag[elem] = acc_imag;
"""

if _METAL_AVAILABLE:
    _dfop_kernel = mx.fast.metal_kernel(
        name="dfop_gather_cmac",
        input_names=["spec_real_pad", "spec_imag_pad", "coef_real", "coef_imag"],
        output_names=["out_real", "out_imag"],
        source=_DFOP_KERNEL_SOURCE,
    )
else:
    _dfop_kernel = None


def df_op_kernel(
    spec_real_pad: mx.array,
    spec_imag_pad: mx.array,
    coef_real: mx.array,
    coef_imag: mx.array,
    output_time: int,
    nb_df: int,
    df_order: int,
    batch_size: int,
) -> Tuple[mx.array, mx.array]:
    """Fused Metal kernel for DfOp gather + complex MAC.

    Args:
        spec_real_pad: Padded spectrum real part, shape ``(batch, time+pad, nb_df)``.
        spec_imag_pad: Padded spectrum imag part, shape ``(batch, time+pad, nb_df)``.
        coef_real: Filter coef real part, shape ``(batch, time, nb_df, df_order)``.
        coef_imag: Filter coef imag part, shape ``(batch, time, nb_df, df_order)``.
        output_time: Number of output time steps.
        nb_df: Number of DF frequency bins.
        df_order: Filter order (number of taps).
        batch_size: Batch dimension size.

    Returns:
        Tuple of ``(out_real, out_imag)``, each ``(batch, time, nb_df)``.

    Raises:
        RuntimeError: If ``mx.fast.metal_kernel`` is not available.
    """
    if _dfop_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is not available")

    total_elements = batch_size * output_time * nb_df
    out_shape = (batch_size, output_time, nb_df)

    outputs = _dfop_kernel(
        inputs=[spec_real_pad, spec_imag_pad, coef_real, coef_imag],
        template=[("T", coef_real.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(min(256, total_elements), 1, 1),
        output_shapes=[out_shape, out_shape],
        output_dtypes=[coef_real.dtype, coef_real.dtype],
    )
    return outputs[0], outputs[1]


# ---------------------------------------------------------------------------
# iSTFT: fused overlap-add + window normalization
# ---------------------------------------------------------------------------

_ISTFT_OLA_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;

    int hop_length = config_arr[0];
    int output_length = config_arr[1];
    int num_frames = frames_shape[1];
    int n_fft_k = frames_shape[2];

    int n = elem % output_length;
    int b = elem / output_length;

    // Determine which frames contribute to output position n
    int first_frame = max(0, (n - n_fft_k + hop_length) / hop_length);
    int last_frame = min(num_frames - 1, n / hop_length);

    T acc = 0;
    for (int i = first_frame; i <= last_frame; i++) {
        int offset = n - i * hop_length;
        if (offset >= 0 && offset < n_fft_k) {
            int frame_idx = b * num_frames * n_fft_k + i * n_fft_k + offset;
            acc += frames[frame_idx];
        }
    }

    // Normalize by cached window norm
    T norm = window_norm[n];
    out[elem] = (norm > T(1e-8)) ? acc / norm : acc;
"""

if _METAL_AVAILABLE:
    _istft_ola_kernel = mx.fast.metal_kernel(
        name="istft_overlap_add",
        input_names=["frames", "window_norm", "config_arr"],
        output_names=["out"],
        source=_ISTFT_OLA_KERNEL_SOURCE,
    )
else:
    _istft_ola_kernel = None


def istft_overlap_add_kernel(
    frames: mx.array,
    window_norm: mx.array,
    hop_length: int,
    output_length: int,
    batch_size: int,
) -> mx.array:
    """Fused Metal kernel for iSTFT overlap-add + window normalization.

    Replaces the Python for-loop overlap-add and separate normalization step
    with a single GPU kernel dispatch.

    Args:
        frames: Windowed IRFFT output, shape ``(batch, num_frames, n_fft)``.
        window_norm: Precomputed window normalization, shape ``(output_length,)``.
        hop_length: Hop size in samples.
        output_length: Length of the output signal (before center-trim).
        batch_size: Batch dimension size.

    Returns:
        Overlap-added and normalized output, shape ``(batch, output_length)``.

    Raises:
        RuntimeError: If ``mx.fast.metal_kernel`` is not available.
    """
    if _istft_ola_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is not available")

    config_arr = mx.array([hop_length, output_length], dtype=mx.int32)
    out_shape = (batch_size, output_length)
    total_elements = batch_size * output_length

    outputs = _istft_ola_kernel(
        inputs=[frames, window_norm, config_arr],
        template=[("T", frames.dtype)],
        grid=(total_elements, 1, 1),
        threadgroup=(min(256, total_elements), 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[frames.dtype],
    )
    return outputs[0]
