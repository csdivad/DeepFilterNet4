import argparse

from df_mlx.hf_paths import hf_dataset_fsspec_path, normalize_hf_dataset_cache_dir
from df_mlx.run_config import RunConfig
from df_mlx.training_cli import _apply_cli_overrides


def test_normalize_hf_dataset_cache_dir_adds_dataset_prefix():
    assert normalize_hf_dataset_cache_dir("hf://sealad886/mlx_datastore") == "hf://datasets/sealad886/mlx_datastore"


def test_normalize_hf_dataset_cache_dir_keeps_explicit_namespace():
    assert (
        normalize_hf_dataset_cache_dir("hf://datasets/sealad886/mlx_datastore")
        == "hf://datasets/sealad886/mlx_datastore"
    )


def test_hf_dataset_fsspec_path_strips_scheme_after_normalization():
    assert hf_dataset_fsspec_path("hf://sealad886/mlx_datastore") == "datasets/sealad886/mlx_datastore"


def test_apply_cli_overrides_normalizes_cache_hf_target_repo():
    cfg = RunConfig()
    args = argparse.Namespace(cache_hf="sealad886/mlx_datastore")

    _apply_cli_overrides(cfg, args, ["--cache-hf", "sealad886/mlx_datastore"])

    assert cfg.dataset.cache_dir == "hf://datasets/sealad886/mlx_datastore"
