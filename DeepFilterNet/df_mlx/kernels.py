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
