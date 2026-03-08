#!/usr/bin/env python3
"""Convert the shipped DeepFilterNet3 archive into an MLX model directory.

This extracts ``models/DeepFilterNet3.zip``, converts the PyTorch checkpoint to
MLX ``safetensors``, copies the original ``config.ini``, and validates the
result with a strict MLX weight load plus a dummy forward pass.
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import mlx.core as mx

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "DeepFilterNet"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_tuple(value: str) -> tuple[int, int]:
    left, right = (part.strip() for part in value.split(",", maxsplit=1))
    return int(left), int(right)


def _build_model_params(config_path: Path):
    from df.spectral import compute_erb_fb
    from df_mlx.deepfilternet3 import ModelParams3

    parser = configparser.ConfigParser()
    parser.read(config_path)

    sec_df = parser["df"]
    sec_model = parser["deepfilternet"]

    params = ModelParams3()
    params.sr = sec_df.getint("sr")
    params.fft_size = sec_df.getint("fft_size")
    params.hop_size = sec_df.getint("hop_size")
    params.nb_erb = sec_df.getint("nb_erb")
    params.nb_df = sec_df.getint("nb_df")
    params.df_order = sec_df.getint("df_order")
    params.df_lookahead = sec_df.getint("df_lookahead")
    params.lsnr_min = sec_df.getfloat("lsnr_min")
    params.lsnr_max = sec_df.getfloat("lsnr_max")
    params.erb_widths = compute_erb_fb(
        sr=params.sr,
        fft_size=params.fft_size,
        nb_bands=params.nb_erb,
        min_nb_freqs=sec_df.getint("min_nb_erb_freqs"),
    )

    params.conv_lookahead = sec_model.getint("conv_lookahead")
    params.conv_ch = sec_model.getint("conv_ch")
    params.conv_depthwise = sec_model.getboolean("conv_depthwise", fallback=True)
    params.convt_depthwise = sec_model.getboolean("convt_depthwise", fallback=True)
    params.conv_kernel = _parse_tuple(sec_model.get("conv_kernel"))
    params.convt_kernel = _parse_tuple(sec_model.get("convt_kernel"))
    params.conv_kernel_inp = _parse_tuple(sec_model.get("conv_kernel_inp"))
    params.emb_hidden_dim = sec_model.getint("emb_hidden_dim")
    params.emb_num_layers = sec_model.getint("emb_num_layers", fallback=params.emb_num_layers)
    params.emb_gru_skip_enc = sec_model.get("emb_gru_skip_enc", fallback="none")
    params.emb_gru_skip = sec_model.get("emb_gru_skip")
    params.df_hidden_dim = sec_model.getint("df_hidden_dim")
    params.df_num_layers = sec_model.getint("df_num_layers", fallback=params.df_num_layers)
    params.df_gru_skip = sec_model.get("df_gru_skip")
    params.df_pathway_kernel_size_t = sec_model.getint("df_pathway_kernel_size_t")
    params.enc_concat = sec_model.getboolean("enc_concat")
    params.linear_groups = sec_model.getint("linear_groups")
    params.enc_linear_groups = sec_model.getint("enc_linear_groups")
    params.mask_pf = sec_model.getboolean("mask_pf")
    return params


def _build_model(params):
    from df_mlx.deepfilternet3 import DFNet3
    from df_mlx.ops import erb_fb_and_inverse

    erb_fb_matrix, erb_inv_fb = erb_fb_and_inverse(
        sr=params.sr,
        fft_size=params.fft_size,
        nb_bands=params.nb_erb,
        min_width=min(params.erb_widths),
    )
    return DFNet3(erb_fb_matrix, erb_inv_fb, run_df=True, p=params)


def convert_archive(archive_path: Path, output_dir: Path) -> dict[str, object]:
    from df_mlx.convert import load_pytorch_checkpoint

    with tempfile.TemporaryDirectory(prefix="dfn3_mlx_convert_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(temp_dir)

        source_root = temp_dir / "DeepFilterNet3"
        config_path = source_root / "config.ini"
        checkpoint_path = source_root / "checkpoints" / "model_120.ckpt.best"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.ini in extracted archive: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint in extracted archive: {checkpoint_path}")

        params = _build_model_params(config_path)
        params.convt_depthwise = True
        model = _build_model(params)
        mlx_weights, metadata = load_pytorch_checkpoint(checkpoint_path, model_type="dfnet3")

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_checkpoint = checkpoint_dir / "model_120.safetensors"

        shutil.copy2(config_path, output_dir / "config.ini")
        mx.save_safetensors(
            str(output_checkpoint),
            mlx_weights,
            metadata={key: str(value) for key, value in metadata.items()},
        )
        model.load_weights(list(mlx_weights.items()), strict=True)

        batch = 1
        time_steps = 4
        freq_bins = params.fft_size // 2 + 1
        spec_real = mx.random.normal((batch, time_steps, freq_bins))
        spec_imag = mx.random.normal((batch, time_steps, freq_bins))
        feat_erb = mx.random.normal((batch, time_steps, params.nb_erb))
        feat_spec = mx.random.normal((batch, time_steps, params.nb_df, 2))
        out_real, out_imag = model((spec_real, spec_imag), feat_erb, feat_spec, training=False)
        mx.eval(out_real, out_imag)

        report = {
            "archive": str(archive_path),
            "source_checkpoint": str(checkpoint_path),
            "output_checkpoint": str(output_checkpoint),
            "converted_keys": len(mlx_weights),
            "output_shapes": {
                "real": list(out_real.shape),
                "imag": list(out_imag.shape),
            },
        }
        (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=repo_root / "models" / "DeepFilterNet3.zip",
        help="Path to the source DeepFilterNet3 zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "models" / "mlx" / "DeepFilterNet3-MLX",
        help="Directory where the converted MLX model directory will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = convert_archive(args.archive.resolve(), args.output_dir.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
