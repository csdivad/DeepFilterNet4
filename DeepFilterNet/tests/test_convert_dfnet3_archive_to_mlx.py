from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "DeepFilterNet"
SCRIPT_PATH = REPO_ROOT / "scripts" / "mlx" / "convert_dfnet3_archive_to_mlx.py"


def _load_module():
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    spec = importlib.util.spec_from_file_location("convert_dfnet3_archive_to_mlx_test_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_archive_creates_valid_mlx_bundle(tmp_path: Path) -> None:
    module = _load_module()

    archive = REPO_ROOT / "models" / "DeepFilterNet3.zip"
    output_dir = tmp_path / "DeepFilterNet3-MLX"

    report = module.convert_archive(archive, output_dir)

    assert report["archive"] == str(archive)
    assert report["converted_keys"] == 114
    assert report["output_shapes"] == {"real": [1, 4, 481], "imag": [1, 4, 481]}
    assert (output_dir / "config.ini").exists()
    assert (output_dir / "checkpoints" / "model_120.safetensors").exists()
    assert (output_dir / "conversion_report.json").exists()
