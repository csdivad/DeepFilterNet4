"""Tests for the GAN-phase compile experiment feature flag and guardrail constants."""

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import pytest

from df_mlx.run_config import RunConfig, apply_run_config_dict, generate_run_config_example

# ---------------------------------------------------------------------------
# Guardrail constants (defined here as the canonical reference; the experiment
# implementation will import or mirror these values).
# ---------------------------------------------------------------------------

LOSS_DIVERGENCE_FACTOR: float = 10.0
GRAD_NORM_EXPLOSION_THRESHOLD: float = 100.0
GRAD_NORM_EXPLOSION_WINDOW: int = 5
DISC_ACCURACY_LOW: float = 0.10
DISC_ACCURACY_HIGH: float = 0.90
CONVERGENCE_TOLERANCE: float = 0.05
THROUGHPUT_MIN_RATIO: float = 0.80
PESQ_SISDR_MAX_DROP: float = 0.10

EXPERIMENT_VARIANTS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Feature flag defaults
# ---------------------------------------------------------------------------


class TestFeatureFlagDefault:
    def test_experimental_compile_defaults_false(self):
        cfg = RunConfig()
        assert cfg.gan.experimental_compile is False

    def test_gan_enabled_defaults_false(self):
        cfg = RunConfig()
        assert cfg.gan.enabled is False


# ---------------------------------------------------------------------------
# TOML round-trip
# ---------------------------------------------------------------------------


class TestFeatureFlagToml:
    def test_parse_experimental_compile_true(self):
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"gan": {"experimental_compile": True}})
        assert cfg.gan.experimental_compile is True

    def test_parse_experimental_compile_false(self):
        cfg = RunConfig()
        apply_run_config_dict(cfg, {"gan": {"experimental_compile": False}})
        assert cfg.gan.experimental_compile is False

    def test_parse_experimental_compile_rejects_string(self):
        cfg = RunConfig()
        with pytest.raises((TypeError, ValueError)):
            apply_run_config_dict(cfg, {"gan": {"experimental_compile": "true"}})

    def test_generated_example_includes_experimental_compile(self):
        text = generate_run_config_example()
        data = tomllib.loads(text)
        assert "gan" in data
        assert "experimental_compile" in data["gan"]
        assert data["gan"]["experimental_compile"] is False

    def test_roundtrip_preserves_default(self):
        text = generate_run_config_example()
        data = tomllib.loads(text)
        cfg = RunConfig()
        apply_run_config_dict(cfg, data)
        assert cfg.gan.experimental_compile is False


# ---------------------------------------------------------------------------
# Eager-only GAN path preserved when flag is off
# ---------------------------------------------------------------------------


class TestEagerPathPreserved:
    def test_flag_off_does_not_change_other_gan_defaults(self):
        cfg = RunConfig()
        assert cfg.gan.experimental_compile is False
        assert cfg.gan.enabled is False
        assert cfg.gan.start_epoch == 0
        assert cfg.gan.adv_weight == 0.0
        assert cfg.gan.fm_weight == 0.0
        assert cfg.gan.disc_update_freq == 1

    def test_enabling_gan_without_compile_flag(self):
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "gan": {
                    "enabled": True,
                    "start_epoch": 5,
                    "adv_weight": 0.1,
                }
            },
        )
        assert cfg.gan.enabled is True
        assert cfg.gan.experimental_compile is False
        assert cfg.gan.start_epoch == 5


# ---------------------------------------------------------------------------
# Abort criteria constants
# ---------------------------------------------------------------------------


class TestAbortCriteriaConstants:
    def test_loss_divergence_factor(self):
        assert LOSS_DIVERGENCE_FACTOR == 10.0

    def test_grad_norm_explosion_threshold(self):
        assert GRAD_NORM_EXPLOSION_THRESHOLD == 100.0

    def test_grad_norm_explosion_window(self):
        assert GRAD_NORM_EXPLOSION_WINDOW == 5

    def test_disc_accuracy_low(self):
        assert DISC_ACCURACY_LOW == pytest.approx(0.10)

    def test_disc_accuracy_high(self):
        assert DISC_ACCURACY_HIGH == pytest.approx(0.90)

    def test_convergence_tolerance(self):
        assert CONVERGENCE_TOLERANCE == pytest.approx(0.05)

    def test_throughput_min_ratio(self):
        assert THROUGHPUT_MIN_RATIO == pytest.approx(0.80)

    def test_pesq_sisdr_max_drop(self):
        assert PESQ_SISDR_MAX_DROP == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------


class TestExperimentMatrix:
    def test_variant_count(self):
        assert len(EXPERIMENT_VARIANTS) == 4

    def test_variant_labels(self):
        assert EXPERIMENT_VARIANTS == ["A", "B", "C", "D"]

    def test_baseline_is_variant_a(self):
        assert EXPERIMENT_VARIANTS[0] == "A"

    def test_matrix_generation(self):
        matrix = _build_experiment_matrix()
        assert len(matrix) == 4
        assert all("variant" in entry for entry in matrix)
        assert all("generator" in entry for entry in matrix)
        assert all("discriminator" in entry for entry in matrix)
        assert all("compile_scope" in entry for entry in matrix)
        assert all("risk_level" in entry for entry in matrix)

    def test_baseline_variant_has_no_compilation(self):
        matrix = _build_experiment_matrix()
        baseline = [e for e in matrix if e["variant"] == "A"][0]
        assert baseline["compile_scope"] == "none"
        assert baseline["risk_level"] == "none"

    def test_gen_only_variant(self):
        matrix = _build_experiment_matrix()
        gen_only = [e for e in matrix if e["variant"] == "B"][0]
        assert gen_only["generator"] == "compiled"
        assert gen_only["discriminator"] == "eager"
        assert gen_only["risk_level"] == "medium"

    def test_full_compiled_variant(self):
        matrix = _build_experiment_matrix()
        full = [e for e in matrix if e["variant"] == "C"][0]
        assert full["generator"] == "compiled"
        assert full["discriminator"] == "compiled"
        assert full["risk_level"] == "high"

    def test_alternating_variant(self):
        matrix = _build_experiment_matrix()
        alt = [e for e in matrix if e["variant"] == "D"][0]
        assert alt["generator"] == "compiled"
        assert alt["discriminator"] == "eager"
        assert alt["risk_level"] == "medium"


# ---------------------------------------------------------------------------
# Helper: experiment matrix builder
# ---------------------------------------------------------------------------


def _build_experiment_matrix() -> list[dict[str, str]]:
    return [
        {
            "variant": "A",
            "generator": "eager",
            "discriminator": "eager",
            "compile_scope": "none",
            "risk_level": "none",
        },
        {
            "variant": "B",
            "generator": "compiled",
            "discriminator": "eager",
            "compile_scope": "generator_loss_grad",
            "risk_level": "medium",
        },
        {
            "variant": "C",
            "generator": "compiled",
            "discriminator": "compiled",
            "compile_scope": "both",
            "risk_level": "high",
        },
        {
            "variant": "D",
            "generator": "compiled",
            "discriminator": "eager",
            "compile_scope": "generator_steps_only",
            "risk_level": "medium",
        },
    ]


# ---------------------------------------------------------------------------
# GAN-P6/P7/P8 config surface tests
# ---------------------------------------------------------------------------


class TestDiscUpdateFreq:
    """Tests for GAN discriminator update frequency configuration."""

    def test_disc_update_freq_config_default(self):
        """disc_update_freq defaults to 1 in GANConfig."""
        from df_mlx.run_config import GanConfig

        cfg = GanConfig()
        assert cfg.disc_update_freq == 1

    def test_disc_update_freq_skip_logic(self):
        """With freq=2, disc updates happen on even steps only."""
        freq = 2
        results = []
        for step in range(10):
            do_update = (step % freq) == 0
            results.append(do_update)
        # Steps 0, 2, 4, 6, 8 should update
        assert results == [True, False, True, False, True, False, True, False, True, False]

    def test_disc_update_freq_config_fields_exist(self):
        """New config fields exist with correct defaults."""
        from df_mlx.run_config import GanConfig

        cfg = GanConfig()
        assert hasattr(cfg, "cache_gen_waveforms")
        assert cfg.cache_gen_waveforms is False
        assert hasattr(cfg, "disc_gradient_checkpoint")
        assert cfg.disc_gradient_checkpoint is False
        assert hasattr(cfg, "single_eval")
        assert cfg.single_eval is False

    def test_new_fields_in_run_config(self):
        """New fields accessible via RunConfig.gan."""
        cfg = RunConfig()
        assert cfg.gan.cache_gen_waveforms is False
        assert cfg.gan.disc_gradient_checkpoint is False
        assert cfg.gan.single_eval is False

    def test_apply_new_fields_via_dict(self):
        """New fields can be set via apply_run_config_dict."""
        cfg = RunConfig()
        apply_run_config_dict(
            cfg,
            {
                "gan": {
                    "cache_gen_waveforms": True,
                    "disc_gradient_checkpoint": True,
                    "single_eval": True,
                }
            },
        )
        assert cfg.gan.cache_gen_waveforms is True
        assert cfg.gan.disc_gradient_checkpoint is True
        assert cfg.gan.single_eval is True

    def test_generated_example_includes_new_fields(self):
        """Generated TOML example includes new GAN fields."""
        text = generate_run_config_example()
        data = tomllib.loads(text)
        assert "gan" in data
        assert "cache_gen_waveforms" in data["gan"]
        assert data["gan"]["cache_gen_waveforms"] is False
        assert "disc_gradient_checkpoint" in data["gan"]
        assert data["gan"]["disc_gradient_checkpoint"] is False
        assert "single_eval" in data["gan"]
        assert data["gan"]["single_eval"] is False
