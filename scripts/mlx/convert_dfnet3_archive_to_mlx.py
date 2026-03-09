#!/usr/bin/env python3
"""Convert the shipped DeepFilterNet3 archive into an MLX model directory.

This extracts ``models/DeepFilterNet3.zip``, converts the PyTorch checkpoint to
MLX ``safetensors``, copies the original ``config.ini``, and validates the
result with a strict MLX weight load plus a dummy forward pass.
"""

from __future__ import annotations

import argparse
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


def _build_model_params(config_path: Path):
    from df_mlx.deepfilternet3 import load_dfnet3_config

    return load_dfnet3_config(config_path)


def _build_model(params):
    from df_mlx.deepfilternet3 import build_dfnet3_model

    return build_dfnet3_model(params, run_df=True)


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
