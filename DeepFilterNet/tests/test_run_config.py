try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from df_mlx.run_config import (
    RunConfig,
    apply_run_config_dict,
    generate_run_config_example,
    load_run_config,
    set_by_path,
)


def test_print_run_config_parses_to_defaults():
    text = generate_run_config_example()
    data = tomllib.loads(text)
    cfg = RunConfig()
    apply_run_config_dict(cfg, data)
    assert cfg == RunConfig()


def test_run_config_precedence_cli_wins():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"training": {"learning_rate": 1e-4}})
    set_by_path(cfg, "training.learning_rate", 3e-5)
    assert cfg.training.learning_rate == 3e-5


def test_unknown_key_errors_with_suggestion():
    cfg = RunConfig()
    try:
        apply_run_config_dict(cfg, {"trainng": {"epochs": 2}})
    except ValueError as exc:
        msg = str(exc)
        assert "Unknown key" in msg
        assert "training" in msg
    else:
        raise AssertionError("Expected ValueError for unknown key")


def test_load_run_config_roundtrip(tmp_path):
    text = generate_run_config_example()
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_run_config(path)
    assert cfg == RunConfig()


def test_run_config_accepts_embedded_train_ini_tables():
    cfg = RunConfig()
    apply_run_config_dict(
        cfg,
        {
            "train_ini": {
                "df": {"sr": 44100, "fft_size": 1024},
                "train": {"max_epochs": 12},
                "MultiResSpecLoss": {"factor": 0.5},
            }
        },
    )
    assert cfg.train_ini["df"]["sr"] == 44100
    assert cfg.train_ini["df"]["fft_size"] == 1024
    assert cfg.train_ini["train"]["max_epochs"] == 12
    assert cfg.train_ini["MultiResSpecLoss"]["factor"] == 0.5


def test_run_config_accepts_pipeline_awesome_dynamic_loss():
    cfg = RunConfig()
    apply_run_config_dict(cfg, {"loss": {"dynamic_loss": "pipeline_awesome"}})
    assert cfg.loss.dynamic_loss == "pipeline_awesome"
