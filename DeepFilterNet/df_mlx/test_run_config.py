from df_mlx.run_config import RunConfig, apply_run_config_dict


def test_run_config_accepts_pipeline_awesome():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"loss": {"dynamic_loss": "pipeline_awesome"}})
    assert cfg.loss.dynamic_loss == "pipeline_awesome"
