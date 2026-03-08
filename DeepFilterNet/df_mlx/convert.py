"""PyTorch to MLX checkpoint conversion utilities.

This module provides utilities for converting PyTorch DeepFilterNet checkpoints
to MLX format, handling differences in:
- Convolution weight layout (NCHW → NHWC)
- GRU weight naming and structure
- BatchNorm naming conventions
- Module hierarchy differences

Supported models:
- DFNet1 (original GroupedGRU model)
- DFNet2 (multi-layer GroupedGRU/SqueezedGRU)
- DFNet3 (SqueezedGRU_S model)
- DFNet4 (Mamba-based model, limited support)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _first_present(pytorch_state: Dict[str, np.ndarray], *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        value = pytorch_state.get(key)
        if value is not None:
            return value
    return None


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose convolution weight from PyTorch to MLX format.

    PyTorch: (out_ch, in_ch, H, W) - OIHW
    MLX: (out_ch, H, W, in_ch) - OHWI

    Args:
        weight: PyTorch convolution weight (4D array)

    Returns:
        MLX-format convolution weight
    """
    if weight.ndim != 4:
        return weight
    return np.transpose(weight, (0, 2, 3, 1))


def transpose_conv_transpose_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose transposed convolution weight from PyTorch to MLX format.

    PyTorch ConvTranspose2d: (in_ch, out_ch, H, W) - IOHW
    MLX ConvTranspose2d: (out_ch, H, W, in_ch) - OHWI

    The key difference is that PyTorch has (in_ch, out_ch) while
    we need (out_ch, ..., in_ch) for MLX.
    """
    if weight.ndim != 4:
        return weight
    if weight.shape[1] == 1 and weight.shape[0] > 1:
        return np.transpose(weight, (0, 2, 3, 1))
    # PyTorch (in, out, H, W) → MLX (out, H, W, in)
    return np.transpose(weight, (1, 2, 3, 0))


def convert_gru_weights(
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: Optional[np.ndarray] = None,
    bias_hh: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Convert PyTorch GRU weights to MLX format.

    PyTorch GRU stores weights as:
    - weight_ih: (3*hidden, input) - gates: reset, update, new
    - weight_hh: (3*hidden, hidden)
    - bias_ih: (3*hidden,)
    - bias_hh: (3*hidden,)

    MLX GRU expects:
    - Wx: (3*hidden, input) - gates: reset, update, new
    - Wh: (3*hidden, hidden)
    - b: (3*hidden,) - combined bias_ih + bias_hh for r, z gates
    - bhn: (hidden,) - separate hidden bias for n gate
    """
    hidden_size = weight_hh.shape[1]

    # Transpose weights: PyTorch (out, in) → MLX (out, in) (same for GRU)
    Wx = weight_ih.T  # (input, 3*hidden) → MLX expects (3*hidden, input)
    Wh = weight_hh.T  # (hidden, 3*hidden) → MLX expects (3*hidden, hidden)

    # Actually MLX GRU uses (out, in) format
    Wx = weight_ih  # Keep as (3*hidden, input)
    Wh = weight_hh  # Keep as (3*hidden, hidden)

    # Combine biases
    # For r and z gates, combine input and hidden biases
    # For n gate, keep hidden bias separate (for numerical stability)
    if bias_ih is not None and bias_hh is not None:
        # Split biases by gate
        bi_r, bi_z, bi_n = np.split(bias_ih, 3)
        bh_r, bh_z, bh_n = np.split(bias_hh, 3)

        # Combined bias for r, z, n (input part)
        b = np.concatenate([bi_r + bh_r, bi_z + bh_z, bi_n])
        # Separate hidden bias for n gate
        bhn = bh_n
    else:
        b = np.zeros(3 * hidden_size)
        bhn = np.zeros(hidden_size)

    return {"Wx": Wx, "Wh": Wh, "b": b, "bhn": bhn}


def convert_squeezed_gru_weights(
    pytorch_state: Dict[str, np.ndarray],
    prefix: str,
    mlx_prefix: str,
) -> Dict[str, np.ndarray]:
    """Convert SqueezedGRU weights from PyTorch to MLX.

    SqueezedGRU has:
    - linear_in: input projection
    - gru: the actual GRU
    - linear_out: output projection (optional)
    """
    result = {}

    # Linear input
    lin_in_w = _first_present(pytorch_state, f"{prefix}.linear_in.weight", f"{prefix}.linear_in.0.weight")
    lin_in_b = _first_present(pytorch_state, f"{prefix}.linear_in.bias", f"{prefix}.linear_in.0.bias")
    if lin_in_w is not None:
        if lin_in_w.ndim == 3:
            result[f"{mlx_prefix}.linear_in.weight"] = lin_in_w
        else:
            result[f"{mlx_prefix}.linear_in.weight"] = lin_in_w.T.reshape(1, -1, lin_in_w.shape[0])
        if lin_in_b is not None:
            result[f"{mlx_prefix}.linear_in.bias"] = lin_in_b

    # GRU layers
    layer_idx = 0
    while True:
        gru_ih = pytorch_state.get(f"{prefix}.gru.weight_ih_l{layer_idx}")
        gru_hh = pytorch_state.get(f"{prefix}.gru.weight_hh_l{layer_idx}")
        if gru_ih is None or gru_hh is None:
            break
        gru_bih = pytorch_state.get(f"{prefix}.gru.bias_ih_l{layer_idx}")
        gru_bhh = pytorch_state.get(f"{prefix}.gru.bias_hh_l{layer_idx}")
        gru_weights = convert_gru_weights(gru_ih, gru_hh, gru_bih, gru_bhh)
        target_prefix = f"{mlx_prefix}.gru" if layer_idx == 0 else f"{mlx_prefix}.gru_layers.{layer_idx - 1}"
        for k, v in gru_weights.items():
            result[f"{target_prefix}.{k}"] = v
        layer_idx += 1

    # Linear output (if exists)
    lin_out_w = _first_present(pytorch_state, f"{prefix}.linear_out.weight", f"{prefix}.linear_out.0.weight")
    lin_out_b = _first_present(pytorch_state, f"{prefix}.linear_out.bias", f"{prefix}.linear_out.0.bias")
    if lin_out_w is not None:
        if lin_out_w.ndim == 3:
            result[f"{mlx_prefix}.linear_out.weight"] = lin_out_w
        else:
            result[f"{mlx_prefix}.linear_out.weight"] = lin_out_w.T.reshape(1, -1, lin_out_w.shape[0])
        if lin_out_b is not None:
            result[f"{mlx_prefix}.linear_out.bias"] = lin_out_b

    return result


def convert_df3_conv_block(
    pytorch_state: Dict[str, np.ndarray],
    prefix: str,
    mlx_prefix: str,
    *,
    conv_index: int,
    norm_index: Optional[int],
    pointwise_index: Optional[int] = None,
    is_transposed: bool = False,
) -> Dict[str, np.ndarray]:
    result = {}

    conv_w = pytorch_state.get(f"{prefix}.{conv_index}.weight")
    conv_b = pytorch_state.get(f"{prefix}.{conv_index}.bias")
    if conv_w is not None:
        if is_transposed:
            result[f"{mlx_prefix}.conv.weight"] = transpose_conv_transpose_weight(conv_w)
        else:
            result[f"{mlx_prefix}.conv.weight"] = transpose_conv_weight(conv_w)
    if conv_b is not None:
        result[f"{mlx_prefix}.conv.bias"] = conv_b

    if pointwise_index is not None:
        pointwise_w = pytorch_state.get(f"{prefix}.{pointwise_index}.weight")
        if pointwise_w is not None:
            result[f"{mlx_prefix}.pointwise_conv.weight"] = transpose_conv_weight(pointwise_w)

    if norm_index is not None:
        bn_prefix = f"{prefix}.{norm_index}"
        bn_w = pytorch_state.get(f"{bn_prefix}.weight")
        if bn_w is not None:
            result[f"{mlx_prefix}.norm_layer.weight"] = bn_w
            result[f"{mlx_prefix}.norm_layer.bias"] = pytorch_state.get(f"{bn_prefix}.bias")
            result[f"{mlx_prefix}.norm_layer.running_mean"] = pytorch_state.get(f"{bn_prefix}.running_mean")
            result[f"{mlx_prefix}.norm_layer.running_var"] = pytorch_state.get(f"{bn_prefix}.running_var")

    return result


def convert_conv2d_norm_act(
    pytorch_state: Dict[str, np.ndarray],
    prefix: str,
    mlx_prefix: str,
    is_transposed: bool = False,
) -> Dict[str, np.ndarray]:
    """Convert Conv2dNormAct weights from PyTorch to MLX."""
    result = {}

    # Convolution weight
    conv_w = pytorch_state.get(f"{prefix}.conv.weight")
    if conv_w is None:
        # Try alternate naming
        conv_w = pytorch_state.get(f"{prefix}.0.weight")
    if conv_w is not None:
        if is_transposed:
            result[f"{mlx_prefix}.conv.weight"] = transpose_conv_transpose_weight(conv_w)
        else:
            result[f"{mlx_prefix}.conv.weight"] = transpose_conv_weight(conv_w)

    conv_b = pytorch_state.get(f"{prefix}.conv.bias")
    if conv_b is None:
        conv_b = pytorch_state.get(f"{prefix}.0.bias")
    if conv_b is not None:
        result[f"{mlx_prefix}.conv.bias"] = conv_b

    # BatchNorm
    for bn_suffix in ["norm", "bn", "1"]:
        bn_w = pytorch_state.get(f"{prefix}.{bn_suffix}.weight")
        if bn_w is not None:
            result[f"{mlx_prefix}.norm_layer.weight"] = bn_w
            result[f"{mlx_prefix}.norm_layer.bias"] = pytorch_state.get(f"{prefix}.{bn_suffix}.bias")
            result[f"{mlx_prefix}.norm_layer.running_mean"] = pytorch_state.get(f"{prefix}.{bn_suffix}.running_mean")
            result[f"{mlx_prefix}.norm_layer.running_var"] = pytorch_state.get(f"{prefix}.{bn_suffix}.running_var")
            break

    return result


def convert_grouped_linear(
    pytorch_state: Dict[str, np.ndarray],
    prefix: str,
    mlx_prefix: str,
    num_groups: int = 1,
) -> Dict[str, np.ndarray]:
    """Convert GroupedLinear weights from PyTorch to MLX.

    MLX GroupedLinear expects weight shape: (groups, in_features/groups, out_features)
    """
    result = {}

    weight = _first_present(pytorch_state, f"{prefix}.weight", f"{prefix}.0.weight")
    bias = _first_present(pytorch_state, f"{prefix}.bias", f"{prefix}.0.bias")

    if weight is not None:
        if weight.ndim == 3:
            result[f"{mlx_prefix}.weight"] = weight
        else:
            # PyTorch: (out, in) → MLX GroupedLinear: (groups, in/groups, out)
            out_features, in_features = weight.shape
            group_in = in_features // num_groups
            weight_grouped = weight.reshape(num_groups, out_features // num_groups, group_in)
            weight_grouped = np.transpose(weight_grouped, (0, 2, 1))
            result[f"{mlx_prefix}.weight"] = weight_grouped

    if bias is not None:
        result[f"{mlx_prefix}.bias"] = bias

    return result


def convert_dfnet3_checkpoint(
    pytorch_state: Dict[str, np.ndarray],
) -> Dict[str, mx.array]:
    """Convert DFNet3 PyTorch checkpoint to MLX format.

    Args:
        pytorch_state: PyTorch state dict as numpy arrays

    Returns:
        MLX-compatible weight dictionary
    """
    mlx_state = {}

    encoder_conv_specs = [
        ("enc.erb_conv0", "encoder.erb_conv0", 1, 2, None, False),
        ("enc.erb_conv1", "encoder.erb_conv1", 0, 2, 1, False),
        ("enc.erb_conv2", "encoder.erb_conv2", 0, 2, 1, False),
        ("enc.erb_conv3", "encoder.erb_conv3", 0, 2, 1, False),
        ("enc.df_conv0", "encoder.df_conv0", 1, 3, 2, False),
        ("enc.df_conv1", "encoder.df_conv1", 0, 2, 1, False),
    ]
    for pt_prefix, mlx_prefix, conv_index, norm_index, pointwise_index, is_transposed in encoder_conv_specs:
        mlx_state.update(
            convert_df3_conv_block(
                pytorch_state,
                pt_prefix,
                mlx_prefix,
                conv_index=conv_index,
                norm_index=norm_index,
                pointwise_index=pointwise_index,
                is_transposed=is_transposed,
            )
        )

    # Encoder DF embedding projection
    df_emb_w = pytorch_state.get("enc.df_fc_emb.weight")
    if df_emb_w is not None or pytorch_state.get("enc.df_fc_emb.0.weight") is not None:
        mlx_state.update(convert_grouped_linear(pytorch_state, "enc.df_fc_emb", "encoder.df_fc_emb.layers.0"))

    # Encoder embedding GRU (SqueezedGRU_S)
    mlx_state.update(convert_squeezed_gru_weights(pytorch_state, "enc.emb_gru", "encoder.emb_gru"))

    # LSNR output
    lsnr_w = pytorch_state.get("enc.lsnr_fc.0.weight")
    if lsnr_w is not None:
        mlx_state["encoder.lsnr_fc.layers.0.weight"] = lsnr_w
        mlx_state["encoder.lsnr_fc.layers.0.bias"] = pytorch_state.get("enc.lsnr_fc.0.bias")

    # ERB Decoder GRU
    mlx_state.update(convert_squeezed_gru_weights(pytorch_state, "erb_dec.emb_gru", "erb_decoder.emb_gru"))

    # ERB decoder convolutions
    erb_decoder_specs = [
        ("erb_dec.conv3p", "erb_decoder.conv3p", 0, 1, None, False),
        ("erb_dec.convt3", "erb_decoder.convt3", 0, 2, 1, False),
        ("erb_dec.conv2p", "erb_decoder.conv2p", 0, 1, None, False),
        ("erb_dec.convt2", "erb_decoder.convt2", 0, 2, 1, True),
        ("erb_dec.conv1p", "erb_decoder.conv1p", 0, 1, None, False),
        ("erb_dec.convt1", "erb_decoder.convt1", 0, 2, 1, True),
        ("erb_dec.conv0p", "erb_decoder.conv0p", 0, 1, None, False),
        ("erb_dec.conv0_out", "erb_decoder.conv0_out", 0, None, None, False),
    ]
    for pt_prefix, mlx_prefix, conv_index, norm_index, pointwise_index, is_transposed in erb_decoder_specs:
        mlx_state.update(
            convert_df3_conv_block(
                pytorch_state,
                pt_prefix,
                mlx_prefix,
                conv_index=conv_index,
                norm_index=norm_index,
                pointwise_index=pointwise_index,
                is_transposed=is_transposed,
            )
        )

    # DF Decoder
    mlx_state.update(
        convert_df3_conv_block(
            pytorch_state,
            "df_dec.df_convp",
            "df_decoder.df_convp",
            conv_index=1,
            norm_index=3,
            pointwise_index=2,
            is_transposed=False,
        )
    )

    # DF GRU
    mlx_state.update(convert_squeezed_gru_weights(pytorch_state, "df_dec.df_gru", "df_decoder.df_gru"))

    # DF skip
    mlx_state.update(convert_grouped_linear(pytorch_state, "df_dec.df_skip", "df_decoder.df_skip"))

    # DF output
    df_out_w = _first_present(pytorch_state, "df_dec.df_out.weight", "df_dec.df_out.0.weight")
    if df_out_w is not None:
        mlx_state.update(convert_grouped_linear(pytorch_state, "df_dec.df_out", "df_decoder.df_out.layers.0"))

    # Convert all to mx.array
    return {k: mx.array(v) for k, v in mlx_state.items() if v is not None}


def load_pytorch_checkpoint(
    checkpoint_path: Union[str, Path],
    model_type: str = "dfnet3",
) -> Tuple[Dict[str, mx.array], Dict[str, Any]]:
    """Load and convert a PyTorch checkpoint to MLX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.ckpt, .pth, or .pt)
        model_type: Type of model ("dfnet1", "dfnet2", "dfnet3", "dfnet4")

    Returns:
        Tuple of (mlx_weights, metadata) where metadata contains any
        extra information from the checkpoint (epoch, optimizer state, etc.)
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        metadata = {k: v for k, v in checkpoint.items() if k not in ["model", "state_dict"]}
    else:
        state_dict = checkpoint
        metadata = {}

    # Convert to numpy
    numpy_state = {k: v.numpy() for k, v in state_dict.items()}

    # Convert based on model type
    if model_type.lower() == "dfnet3":
        mlx_weights = convert_dfnet3_checkpoint(numpy_state)
    elif model_type.lower() == "dfnet2":
        mlx_weights = convert_dfnet2_checkpoint(numpy_state)
    elif model_type.lower() == "dfnet1":
        mlx_weights = convert_dfnet1_checkpoint(numpy_state)
    else:
        # Generic conversion (basic transpositions only)
        mlx_weights = convert_generic_checkpoint(numpy_state)

    return mlx_weights, metadata


def convert_dfnet2_checkpoint(
    pytorch_state: Dict[str, np.ndarray],
) -> Dict[str, mx.array]:
    """Convert DFNet2 PyTorch checkpoint to MLX format."""
    # DFNet2 has similar structure to DFNet3 but uses GroupedGRU
    # This is a simplified version - full implementation would need
    # to handle the grouped GRU layers
    mlx_state = {}

    # Basic conversion - similar to DFNet3 but with different GRU handling
    # TODO: Implement full DFNet2 conversion with multi-layer GRU support

    return {k: mx.array(v) for k, v in mlx_state.items() if v is not None}


def convert_dfnet1_checkpoint(
    pytorch_state: Dict[str, np.ndarray],
) -> Dict[str, mx.array]:
    """Convert DFNet1 PyTorch checkpoint to MLX format."""
    # DFNet1 is the original architecture with GroupedGRU
    # TODO: Implement full DFNet1 conversion

    mlx_state = {}
    return {k: mx.array(v) for k, v in mlx_state.items() if v is not None}


def convert_generic_checkpoint(
    pytorch_state: Dict[str, np.ndarray],
) -> Dict[str, mx.array]:
    """Generic checkpoint conversion with basic transformations.

    Handles:
    - Conv2d weight transposition
    - ConvTranspose2d weight transposition
    - Basic renaming
    """
    mlx_state = {}

    for name, param in pytorch_state.items():
        mlx_param = param

        # Handle convolution weights
        if "conv" in name.lower() and "weight" in name and param.ndim == 4:
            if "convt" in name.lower() or "conv_transpose" in name.lower():
                mlx_param = transpose_conv_transpose_weight(param)
            else:
                mlx_param = transpose_conv_weight(param)

        mlx_state[name] = mx.array(mlx_param)

    return mlx_state


def save_mlx_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save MLX model checkpoint.

    Args:
        model: MLX model to save
        path: Output path (.safetensors recommended)
        metadata: Optional metadata to include
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get flattened weights using tree_flatten
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}.{i}", sep=sep).items())
            elif isinstance(v, mx.array):
                items.append((new_key, v))
        return dict(items)

    weights = flatten_dict(model.parameters())

    # Save using safetensors format
    if path.suffix == ".safetensors":
        mx.save_safetensors(str(path), weights, metadata=metadata or {})
    else:
        # NPZ format
        mx.savez(str(path), **weights)

    print(f"Saved MLX checkpoint: {path}")


def load_mlx_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    strict: bool = True,
) -> nn.Module:
    """Load MLX model checkpoint.

    Args:
        model: MLX model to load weights into
        path: Path to checkpoint (.safetensors or .npz)
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Model with loaded weights
    """
    path = Path(path)

    # Load weights
    weights = mx.load(str(path))

    # Convert to list of tuples for load_weights
    if isinstance(weights, dict):
        weight_list = list(weights.items())
    else:
        # Handle tuple return from some load calls
        weight_list = list(weights[0].items()) if isinstance(weights, tuple) else weights

    # Load into model
    model.load_weights(weight_list, strict=strict)

    print(f"Loaded MLX checkpoint: {path}")
    return model


# Convenience functions for common use cases
def convert_and_save(
    pytorch_path: Union[str, Path],
    mlx_path: Union[str, Path],
    model_type: str = "dfnet3",
) -> None:
    """Convert PyTorch checkpoint to MLX and save.

    Args:
        pytorch_path: Path to PyTorch checkpoint
        mlx_path: Output path for MLX checkpoint
        model_type: Model type for conversion
    """
    mlx_weights, metadata = load_pytorch_checkpoint(pytorch_path, model_type)

    mlx_path = Path(mlx_path)
    mlx_path.parent.mkdir(parents=True, exist_ok=True)

    if mlx_path.suffix == ".safetensors":
        mx.save_safetensors(str(mlx_path), mlx_weights, metadata=metadata)
    else:
        mx.savez(str(mlx_path), **mlx_weights)

    print(f"Converted {pytorch_path} -> {mlx_path}")
    print(f"  Keys: {len(mlx_weights)}")


if __name__ == "__main__":
    # Test conversion utilities
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoints to MLX")
    parser.add_argument("input", help="Input PyTorch checkpoint path")
    parser.add_argument("output", help="Output MLX checkpoint path")
    parser.add_argument(
        "--model-type",
        choices=["dfnet1", "dfnet2", "dfnet3", "dfnet4"],
        default="dfnet3",
        help="Model type",
    )

    args = parser.parse_args()
    convert_and_save(args.input, args.output, args.model_type)
