import tempfile
from pathlib import Path

from df_mlx.config import get_default_config
from df_mlx.run_config import RunConfig
from df_mlx.train_dynamic_config import apply_train_ini_config


def test_apply_train_ini_config_maps_values():
    ini = """
[df]
SR = 44100
FFT_SIZE = 1024
HOP_SIZE = 256
NB_ERB = 24
NB_DF = 64
DF_ORDER = 4

[train]
MAX_EPOCHS = 77
BATCH_SIZE = 16
NUM_WORKERS = 6
NUM_PREFETCH_BATCHES = 12
MAX_SAMPLE_LEN_S = 4.0
SEED = 123
DATALOADER_SNRS = -10, 0, 20
DATALOADER_GAINS = -6, 6

[optim]
LR = 1e-5
LR_MIN = 1e-7
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 2

[distortion]
P_REVERB = 0.2
P_CLIPPING = 0.1
P_BANDWIDTH_EXT = 0.05
P_INTERFER_SP = 0.15

[deepfilternet4]
BACKBONE = attention
MODEL_VARIANT = lite
CONV_CH = 32
CONV_KERNEL = 1, 3
EMB_HIDDEN_DIM = 128
EMB_NUM_LAYERS = 3
DF_HIDDEN_DIM = 192
DF_NUM_LAYERS = 2
MASK_PF = true
PF_BETA = 0.05
MAMBA_D_STATE = 8
MAMBA_D_CONV = 2
MAMBA_EXPAND = 3

[loss]
MULTI_RES_STFT_F = 1.2
MULTI_RES_STFT_GAMMA = 0.7

[MultiResSpecLoss]
factor = 0.9
gamma = 0.5
factor_complex = 0.4
fft_sizes = 256, 512, 1024
hop_sizes = 64, 128, 256
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        ini_path = Path(tmpdir) / "config.ini"
        ini_path.write_text(ini)

        run_cfg = RunConfig()
        model_cfg = get_default_config()
        overrides = apply_train_ini_config(str(ini_path), run_cfg, model_cfg)

    assert run_cfg.training.epochs == 77
    assert run_cfg.training.batch_size == 16
    assert run_cfg.training.learning_rate == 1e-5
    assert run_cfg.training.learning_rate_min == 1e-7
    assert run_cfg.training.weight_decay == 0.01
    assert run_cfg.dataloader.num_workers == 6
    assert run_cfg.dataloader.prefetch_size == 12
    assert run_cfg.model.backbone_type == "attention"
    assert run_cfg.model.variant == "lite"

    assert run_cfg.loss.mrstft.factor == 0.9
    assert run_cfg.loss.mrstft.gamma == 0.5
    assert run_cfg.loss.mrstft.f_complex == 0.4
    assert run_cfg.loss.mrstft.fft_sizes == [256, 512, 1024]
    assert run_cfg.loss.mrstft.hop_sizes == [64, 128, 256]

    assert overrides.dataset_overrides["sample_rate"] == 44100
    assert overrides.dataset_overrides["fft_size"] == 1024
    assert overrides.dataset_overrides["hop_size"] == 256
    assert overrides.dataset_overrides["nb_erb"] == 24
    assert overrides.dataset_overrides["nb_df"] == 64
    assert overrides.dataset_overrides["segment_length"] == 4.0
    assert overrides.dataset_overrides["snr_range"] == (-10.0, 20.0)
    assert overrides.dataset_overrides["speech_gain_range"] == (-6.0, 6.0)
    assert overrides.dataset_overrides["noise_gain_range"] == (-6.0, 6.0)
    assert overrides.dataset_overrides["p_reverb"] == 0.2
    assert overrides.dataset_overrides["p_clipping"] == 0.1
    assert overrides.dataset_overrides["p_bandwidth_ext"] == 0.05
    assert overrides.dataset_overrides["p_interfer_speech"] == 0.15

    assert model_cfg.audio.sr == 44100
    assert model_cfg.audio.fft_size == 1024
    assert model_cfg.audio.hop_size == 256
    assert model_cfg.erb.nb_erb == 24
    assert model_cfg.df.nb_df == 64
    assert model_cfg.df.df_order == 4
    assert model_cfg.encoder.conv_channels == 32
    assert model_cfg.encoder.conv_kernel == [1, 3]
    assert model_cfg.encoder.emb_hidden_dim == 128
    assert model_cfg.encoder.num_enc_layers == 3
    assert model_cfg.df.nb_df_hidden == 192
    assert model_cfg.df.df_n_layers == 2
    assert model_cfg.df.mask_pf is True
    assert model_cfg.df.pf_beta == 0.05
    assert model_cfg.backbone.d_state == 8
    assert model_cfg.backbone.d_conv == 2
    assert model_cfg.backbone.expand_factor == 3
    assert overrides.warnings == []
