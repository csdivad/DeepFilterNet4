"""DeepFilterNet3 model implementation for MLX.

This module provides the MLX implementation of DeepFilterNet3,
a GRU-based speech enhancement model. DFNet3 uses SqueezedGRU
instead of Mamba, providing compatibility with older checkpoints.

Architecture:
- Encoder: Parallel ERB and DF pathways with SqueezedGRU
- ERB Decoder: Spectral mask estimation for ERB bands
- DF Decoder: Deep filtering coefficient prediction

Key differences from DFNet4:
- Uses GRU instead of Mamba for temporal modeling
- Different encoder architecture with separate pathways
- Skip connections between encoder and decoder
"""

import configparser
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .modules import Conv2dNormAct as BaseConv2dNormAct
from .modules import ConvTranspose2dNormAct as BaseConvTranspose2dNormAct
from .modules import (
    DfOp,
    GroupedLinear,
    SqueezedGRU_S,
    erb_fb,
)


@dataclass
class ModelParams3:
    """DFNet3 model parameters."""

    # Audio parameters
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    df_order: int = 5
    df_lookahead: int = 0
    norm_tau: float = 1.0
    lsnr_min: float = -15.0
    lsnr_max: float = 40.0
    erb_widths: list = field(default_factory=lambda: [1] * 32)

    # Convolution settings
    conv_lookahead: int = 0
    conv_ch: int = 16
    conv_depthwise: bool = True
    convt_depthwise: bool = True
    conv_kernel: Tuple[int, int] = (1, 3)
    convt_kernel: Tuple[int, int] = (1, 3)
    conv_kernel_inp: Tuple[int, int] = (3, 3)

    # Embedding settings
    emb_hidden_dim: int = 256
    emb_num_layers: int = 2
    emb_gru_skip_enc: str = "none"
    emb_gru_skip: str = "none"

    # DF settings
    df_hidden_dim: int = 256
    df_num_layers: int = 3
    df_gru_skip: str = "none"
    df_pathway_kernel_size_t: int = 1

    # Encoder settings
    enc_concat: bool = False
    linear_groups: int = 1
    enc_linear_groups: int = 16

    # Post-filter settings
    mask_pf: bool = False
    pf_beta: float = 0.02


def _parse_tuple(value: str) -> tuple[int, int]:
    left, right = (part.strip() for part in value.split(",", maxsplit=1))
    return int(left), int(right)


def _freq2erb(freq: float) -> float:
    return 9.265 * math.log(1.0 + freq / (24.7 * 9.265))


def _erb2freq(n_erb: float) -> float:
    return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1.0)


def compute_erb_fb(
    sr: int = 48000,
    fft_size: int = 960,
    nb_bands: int = 32,
    min_nb_freqs: int = 2,
) -> list[int]:
    """Compute ERB band widths matching the original Rust/libDF contract."""

    freq_width = sr / fft_size
    erb_low = _freq2erb(0.0)
    erb_high = _freq2erb(sr / 2)
    step = (erb_high - erb_low) / nb_bands

    erb = [0] * nb_bands
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = _erb2freq(erb_low + i * step)
        fb = round(f / freq_width)
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i - 1] = nb_freqs
        prev_freq = fb

    erb[nb_bands - 1] += 1
    too_large = sum(erb) - (fft_size // 2 + 1)
    if too_large > 0:
        erb[nb_bands - 1] -= too_large

    assert sum(erb) == fft_size // 2 + 1, f"ERB band sum {sum(erb)} != {fft_size // 2 + 1}"
    return erb


def load_dfnet3_config(path: str | Path) -> ModelParams3:
    """Load DFNet3 parameters from a PyTorch-style ``config.ini`` file."""

    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(f"Could not read config file: {path}")

    if "df" not in config:
        raise KeyError(f"Missing [df] section in DFNet3 config: {path}")
    if "deepfilternet" not in config:
        raise KeyError(f"Missing [deepfilternet] section in DFNet3 config: {path}")

    sec_df = config["df"]
    sec_model = config["deepfilternet"]

    params = ModelParams3()
    params.sr = sec_df.getint("sr", fallback=params.sr)
    params.fft_size = sec_df.getint("fft_size", fallback=params.fft_size)
    params.hop_size = sec_df.getint("hop_size", fallback=params.hop_size)
    params.nb_erb = sec_df.getint("nb_erb", fallback=params.nb_erb)
    params.nb_df = sec_df.getint("nb_df", fallback=params.nb_df)
    params.df_order = sec_df.getint("df_order", fallback=params.df_order)
    params.df_lookahead = sec_df.getint("df_lookahead", fallback=params.df_lookahead)
    params.norm_tau = sec_df.getfloat("norm_tau", fallback=params.norm_tau)
    params.lsnr_min = sec_df.getfloat("lsnr_min", fallback=params.lsnr_min)
    params.lsnr_max = sec_df.getfloat("lsnr_max", fallback=params.lsnr_max)
    params.erb_widths = compute_erb_fb(
        sr=params.sr,
        fft_size=params.fft_size,
        nb_bands=params.nb_erb,
        min_nb_freqs=sec_df.getint("min_nb_erb_freqs", fallback=2),
    )

    params.conv_lookahead = sec_model.getint("conv_lookahead", fallback=params.conv_lookahead)
    params.conv_ch = sec_model.getint("conv_ch", fallback=params.conv_ch)
    params.conv_depthwise = sec_model.getboolean("conv_depthwise", fallback=params.conv_depthwise)
    params.convt_depthwise = sec_model.getboolean("convt_depthwise", fallback=params.convt_depthwise)
    if sec_model.get("conv_kernel", fallback=None):
        params.conv_kernel = _parse_tuple(sec_model.get("conv_kernel"))
    if sec_model.get("convt_kernel", fallback=None):
        params.convt_kernel = _parse_tuple(sec_model.get("convt_kernel"))
    if sec_model.get("conv_kernel_inp", fallback=None):
        params.conv_kernel_inp = _parse_tuple(sec_model.get("conv_kernel_inp"))
    params.emb_hidden_dim = sec_model.getint("emb_hidden_dim", fallback=params.emb_hidden_dim)
    params.emb_num_layers = sec_model.getint("emb_num_layers", fallback=params.emb_num_layers)
    params.emb_gru_skip_enc = sec_model.get("emb_gru_skip_enc", fallback=params.emb_gru_skip_enc)
    params.emb_gru_skip = sec_model.get("emb_gru_skip", fallback=params.emb_gru_skip)
    params.df_hidden_dim = sec_model.getint("df_hidden_dim", fallback=params.df_hidden_dim)
    params.df_num_layers = sec_model.getint("df_num_layers", fallback=params.df_num_layers)
    params.df_gru_skip = sec_model.get("df_gru_skip", fallback=params.df_gru_skip)
    params.df_pathway_kernel_size_t = sec_model.getint(
        "df_pathway_kernel_size_t",
        fallback=params.df_pathway_kernel_size_t,
    )
    params.enc_concat = sec_model.getboolean("enc_concat", fallback=params.enc_concat)
    params.linear_groups = sec_model.getint("linear_groups", fallback=params.linear_groups)
    params.enc_linear_groups = sec_model.getint("enc_linear_groups", fallback=params.enc_linear_groups)
    params.mask_pf = sec_model.getboolean("mask_pf", fallback=params.mask_pf)
    params.pf_beta = sec_model.getfloat("pf_beta", fallback=params.pf_beta)

    # The converted MLX DFNet3 checkpoint stores transposed-convolution weights
    # in the separable layout (depthwise + pointwise). Keep the runtime loader
    # aligned with the conversion script so strict MLX weight loading succeeds.
    params.convt_depthwise = True
    return params


def build_dfnet3_model(params: ModelParams3, *, run_df: bool = True) -> "DFNet3":
    """Construct a DFNet3 model from parsed parameters."""

    from .ops import erb_fb_and_inverse

    erb_fb_matrix, erb_inv_fb = erb_fb_and_inverse(
        sr=params.sr,
        fft_size=params.fft_size,
        nb_bands=params.nb_erb,
        min_width=max(1, min(params.erb_widths)),
    )
    return DFNet3(erb_fb_matrix, erb_inv_fb, run_df=run_df, p=params)


def _apply_conv_lookahead(features: mx.array, lookahead: int) -> mx.array:
    """Shift features left by ``lookahead`` frames and zero-pad the tail.

    Mirrors the PyTorch DFNet3 ``ConstantPad2d((0, 0, -lookahead, lookahead), 0)``
    behavior used on feature tensors before the encoder.
    """

    if lookahead <= 0:
        return features

    if lookahead >= features.shape[1]:
        return mx.zeros_like(features)

    shifted = features[:, lookahead:, ...]
    pad_width = [(0, 0), (0, lookahead)] + [(0, 0)] * (features.ndim - 2)
    return mx.pad(shifted, pad_width)


def _as_pair(value: Union[int, Tuple[int, int]]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value


def _causal_time_pad(kernel_size: tuple[int, int], lookahead: int = 0) -> tuple[int, int, int, int]:
    """Return NHWC pad widths for torch-style causal/asymmetric DF3 conv padding."""

    kernel_t, _ = kernel_size
    return (0, 0, max(0, kernel_t - 1 - lookahead), max(0, lookahead))


class Conv2dNormAct(BaseConv2dNormAct):
    """DF3-local conv wrapper matching the original torch time-padding contract."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        norm: Optional[str] = "batch",
        activation: Optional[str] = "relu",
        norm_groups: int = 8,
        separable: bool = False,
        lookahead: int = 0,
    ):
        kernel = _as_pair(kernel_size)
        dilation_pair = _as_pair(dilation)
        freq_pad = kernel[1] // 2 + dilation_pair[1] - 1
        self._time_pad = _causal_time_pad(kernel, lookahead=lookahead)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=(0, freq_pad),
            dilation=dilation_pair,
            bias=bias,
            norm=norm,
            activation=activation,
            norm_groups=norm_groups,
            separable=separable,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if any(self._time_pad):
            x = mx.pad(x, [(0, 0), (self._time_pad[2], self._time_pad[3]), (0, 0), (0, 0)])
        return super().__call__(x)


class ConvTranspose2dNormAct(BaseConvTranspose2dNormAct):
    """DF3-local transposed conv wrapper matching the original torch padding contract."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        norm: Optional[str] = "batch",
        activation: Optional[str] = "relu",
        separable: bool = False,
        lookahead: int = 0,
    ):
        kernel = _as_pair(kernel_size)
        dilation_pair = _as_pair(dilation)
        stride_pair = _as_pair(stride)
        freq_pad = kernel[1] // 2
        self._time_pad = _causal_time_pad(kernel, lookahead=lookahead)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride_pair,
            padding=(kernel[0] - 1, freq_pad + dilation_pair[1] - 1),
            output_padding=(0, freq_pad),
            bias=bias,
            norm=norm,
            activation=activation,
            separable=separable,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if any(self._time_pad):
            x = mx.pad(x, [(0, 0), (self._time_pad[2], self._time_pad[3]), (0, 0), (0, 0)])
        return super().__call__(x)


class Encoder3(nn.Module):
    """DeepFilterNet3 encoder with parallel ERB and DF pathways."""

    def __init__(self, p: ModelParams3):
        super().__init__()
        self.p = p
        conv_ch = p.conv_ch

        # ERB pathway convolutions (NHWC format for MLX)
        # Use padding="same" on stride-2 convs to get clean dimension halving
        self.erb_conv0 = Conv2dNormAct(
            1,
            conv_ch,
            kernel_size=p.conv_kernel_inp,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.erb_conv1 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.conv_kernel,
            stride=(1, 2),
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.erb_conv2 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.conv_kernel,
            stride=(1, 2),
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.erb_conv3 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )

        # DF pathway convolutions
        self.df_conv0 = Conv2dNormAct(
            2,
            conv_ch,
            kernel_size=p.conv_kernel_inp,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.df_conv1 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.conv_kernel,
            stride=(1, 2),
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )

        # Dimension calculations with "same" padding:
        # ERB: nb_erb -> nb_erb//2 -> nb_erb//4 (with same padding, stride-2 halves)
        # DF: nb_df -> nb_df//2
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.emb_out_dim = conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim

        # DF pathway projection to match ERB embedding dimension
        df_emb_in = conv_ch * p.nb_df // 2
        self.df_fc_emb = nn.Sequential(
            GroupedLinear(df_emb_in, self.emb_in_dim, p.enc_linear_groups, bias=False),
            nn.ReLU(),
        )

        # Embedding input dimension
        gru_input_dim = self.emb_in_dim * 2 if p.enc_concat else self.emb_in_dim

        # Embedding GRU
        self.emb_gru = SqueezedGRU_S(
            input_size=gru_input_dim,
            hidden_size=self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            linear_groups=p.linear_groups,
            linear_bias=False,
            gru_skip=(p.emb_gru_skip_enc != "none"),
            linear_act="relu",
        )

        # LSNR estimation
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

        self.enc_concat = p.enc_concat
        self.erb_bins_downsampled = p.nb_erb // 4

    def __call__(
        self, feat_erb: mx.array, feat_spec: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Encode ERB and DF features.

        Args:
            feat_erb: ERB features (batch, time, erb_bins)
            feat_spec: Complex DF features (batch, time, df_bins, 2)

        Returns:
            e0-e3: Encoder skip connections (batch, time, *, conv_ch)
            emb: Embedding (batch, time, emb_out_dim)
            c0: DF pathway output (batch, time, df_bins, conv_ch)
            lsnr: LSNR estimate (batch, time, 1)
        """
        # Expand ERB features: (B, T, F) -> (B, T, F, 1)
        feat_erb = mx.expand_dims(feat_erb, axis=-1)

        # ERB pathway
        e0 = self.erb_conv0(feat_erb)  # (B, T, F, C)
        e1 = self.erb_conv1(e0)  # (B, T, F//2, C)
        e2 = self.erb_conv2(e1)  # (B, T, F//4, C)
        e3 = self.erb_conv3(e2)  # (B, T, F//4, C)

        # DF pathway
        c0 = self.df_conv0(feat_spec)  # (B, T, Fdf, C)
        c1 = self.df_conv1(c0)  # (B, T, Fdf//2, C)

        # Flatten to embeddings
        b, t = e3.shape[:2]
        emb_erb = e3.reshape(b, t, -1)  # (B, T, C*F//4)

        emb_df = c1.reshape(b, t, -1)  # (B, T, C*Fdf//2)
        emb_df = self.df_fc_emb(emb_df)  # (B, T, emb_in_dim)

        # Combine ERB and DF embeddings
        if self.enc_concat:
            emb = mx.concatenate([emb_erb, emb_df], axis=-1)
        else:
            emb = emb_erb + emb_df

        # GRU
        emb, _ = self.emb_gru(emb)

        # LSNR
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder3(nn.Module):
    """DeepFilterNet3 ERB mask decoder."""

    def __init__(self, p: ModelParams3):
        super().__init__()
        conv_ch = p.conv_ch
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim

        # Decoder GRU
        self.emb_gru = SqueezedGRU_S(
            input_size=self.emb_in_dim,
            hidden_size=self.emb_dim,
            output_size=self.emb_in_dim,
            num_layers=max(1, p.emb_num_layers - 1),
            linear_groups=p.linear_groups,
            linear_bias=False,
            gru_skip=(p.emb_gru_skip != "none"),
            linear_act="relu",
        )

        # Decoder convolutions
        # For transposed conv with stride=2, kernel=3: padding=1, output_padding=1 gives exact 2x upsample
        self.conv3p = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=1,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.convt3 = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.conv2p = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=1,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.convt2 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.convt_kernel,
            stride=(1, 2),
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.convt_depthwise,
        )
        self.conv1p = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=1,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.convt1 = ConvTranspose2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=p.convt_kernel,
            stride=(1, 2),
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.convt_depthwise,
        )
        self.conv0p = Conv2dNormAct(
            conv_ch,
            conv_ch,
            kernel_size=1,
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )
        self.conv0_out = Conv2dNormAct(
            conv_ch,
            1,
            kernel_size=p.conv_kernel,
            bias=False,
            norm=None,
            activation="sigmoid",
            separable=p.conv_depthwise,
        )

        self.erb_bins_downsampled = p.nb_erb // 4

    def __call__(self, emb: mx.array, e3: mx.array, e2: mx.array, e1: mx.array, e0: mx.array) -> mx.array:
        """Decode ERB mask.

        Args:
            emb: Embedding (batch, time, emb_dim)
            e0-e3: Encoder skip connections

        Returns:
            ERB mask (batch, time, erb_bins)
        """
        b, t, _ = emb.shape

        # GRU
        emb, _ = self.emb_gru(emb)

        # Reshape to spatial
        emb = emb.reshape(b, t, self.erb_bins_downsampled, -1)

        # Decoder with skip connections
        x = self.conv3p(e3) + emb
        x = self.convt3(x)
        x = self.conv2p(e2) + x
        x = self.convt2(x)
        x = self.conv1p(e1) + x
        x = self.convt1(x)
        x = self.conv0p(e0) + x
        m = self.conv0_out(x)

        # (B, T, F, 1) -> (B, T, F)
        return mx.squeeze(m, axis=-1)


class DfDecoder3(nn.Module):
    """DeepFilterNet3 DF coefficient decoder."""

    def __init__(self, p: ModelParams3):
        super().__init__()
        conv_ch = p.conv_ch
        self.emb_in_dim = conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim
        self.df_bins = p.nb_df
        self.df_order = p.df_order
        self.df_out_ch = p.df_order * 2

        # Pathway convolution
        self.df_convp = Conv2dNormAct(
            conv_ch,
            self.df_out_ch,
            kernel_size=(p.df_pathway_kernel_size_t, 1),
            bias=False,
            norm="batch",
            activation="relu",
            separable=p.conv_depthwise,
        )

        # DF GRU
        self.df_gru = SqueezedGRU_S(
            input_size=self.emb_in_dim,
            hidden_size=self.emb_dim,
            num_layers=p.df_num_layers,
            linear_groups=8,
            linear_bias=False,
            gru_skip=False,
            linear_act="relu",
        )

        # Optional skip
        self.df_skip: Optional[nn.Module] = None
        if p.df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinear(self.emb_in_dim, self.emb_dim, p.linear_groups, bias=False)
        elif p.df_gru_skip == "identity":
            self.df_skip = nn.Identity()

        # Output
        self.df_out = nn.Sequential(
            GroupedLinear(self.emb_dim, self.df_bins * self.df_out_ch, p.linear_groups, bias=False),
            nn.Tanh(),
        )

    def __call__(self, emb: mx.array, c0: mx.array) -> mx.array:
        """Decode DF coefficients.

        Args:
            emb: Embedding (batch, time, emb_dim)
            c0: DF pathway conv output (batch, time, df_bins, conv_ch)

        Returns:
            DF coefficients (batch, time, df_bins, df_order, 2)
        """
        b, t, _ = emb.shape

        # GRU
        c, _ = self.df_gru(emb)

        # Skip
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        # Pathway contribution
        c0_proj = self.df_convp(c0)  # (B, T, F, df_out_ch)

        # Output
        c = self.df_out(c)  # (B, T, F * df_out_ch)
        c = c.reshape(b, t, self.df_bins, self.df_out_ch)

        # Add pathway contribution
        c = c + c0_proj

        # Reshape to (B, T, F, df_order, 2)
        return c.reshape(b, t, self.df_bins, self.df_order, 2)


class DFNet3(nn.Module):
    """DeepFilterNet3 complete model.

    This is a GRU-based alternative to DFNet4 (which uses Mamba).
    It maintains the same input/output interface for compatibility.

    Args:
        erb_fb: ERB filterbank matrix (erb_bins, freq_bins)
        erb_inv_fb: Inverse ERB filterbank matrix
        run_df: Whether to run DF stage
        p: Model parameters
    """

    def __init__(
        self,
        erb_fb_matrix: mx.array,
        erb_inv_fb: mx.array,
        run_df: bool = True,
        p: Optional[ModelParams3] = None,
    ):
        super().__init__()

        if p is None:
            p = ModelParams3()

        self.p = p
        self.run_df = run_df
        self.nb_df = p.nb_df
        self.df_order = p.df_order
        self.nb_erb = p.nb_erb
        self.freq_bins = p.fft_size // 2 + 1

        # Store filterbanks
        self._erb_fb = erb_fb_matrix
        self._erb_inv_fb = erb_inv_fb

        # Model components
        self.encoder = Encoder3(p)
        self.erb_decoder = ErbDecoder3(p)
        self.df_decoder = DfDecoder3(p)

        # DF operation
        self.df_op = DfOp(
            nb_df=p.nb_df,
            df_order=p.df_order,
            df_lookahead=p.df_lookahead,
        )

        # Post-filter settings
        self.post_filter = p.mask_pf
        self.pf_beta = p.pf_beta

    def __call__(
        self,
        spec: Tuple[mx.array, mx.array],
        feat_erb: mx.array,
        feat_spec: mx.array,
        training: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            spec: Input spectrum as (real, imag), each (batch, time, freq)
            feat_erb: ERB features (batch, time, erb_bins)
            feat_spec: DF features (batch, time, df_bins, 2)
            training: Training mode flag

        Returns:
            Enhanced spectrum as (real, imag)
        """
        spec_real, spec_imag = spec

        if self.p.conv_lookahead > 0:
            feat_erb = _apply_conv_lookahead(feat_erb, self.p.conv_lookahead)
            feat_spec = _apply_conv_lookahead(feat_spec, self.p.conv_lookahead)

        # Encode
        e0, e1, e2, e3, emb, c0, lsnr = self.encoder(feat_erb, feat_spec)

        # Decode ERB mask
        erb_mask = self.erb_decoder(emb, e3, e2, e1, e0)  # (B, T, nb_erb)

        # Expand ERB mask to full spectrum
        mask = mx.matmul(erb_mask, self._erb_inv_fb)  # (B, T, freq)

        # Apply mask to non-DF frequencies
        masked_real = spec_real * mask
        masked_imag = spec_imag * mask

        if self.run_df:
            # Decode DF coefficients
            df_coefs = self.df_decoder(emb, c0)

            # Apply DF operation
            df_real, df_imag = self.df_op((spec_real, spec_imag), df_coefs)

            # Combine DF and masked regions
            out_real = mx.concatenate([df_real[:, :, : self.nb_df], masked_real[:, :, self.nb_df :]], axis=-1)
            out_imag = mx.concatenate([df_imag[:, :, : self.nb_df], masked_imag[:, :, self.nb_df :]], axis=-1)
        else:
            out_real, out_imag = masked_real, masked_imag

        # Post-filter
        if self.post_filter:
            out_real, out_imag = self._apply_post_filter((spec_real, spec_imag), (out_real, out_imag))

        return (out_real, out_imag)

    def _apply_post_filter(
        self, orig: Tuple[mx.array, mx.array], enh: Tuple[mx.array, mx.array]
    ) -> Tuple[mx.array, mx.array]:
        """Apply mask-based post-filter."""
        eps = 1e-12
        beta = self.pf_beta

        orig_mag = mx.sqrt(orig[0] ** 2 + orig[1] ** 2 + eps)
        enh_mag = mx.sqrt(enh[0] ** 2 + enh[1] ** 2 + eps)

        mask = mx.clip(enh_mag / orig_mag, eps, 1.0)
        mask_sin = mask * mx.clip(mx.sin(3.14159 * mask / 2), eps, None)
        pf = (1 + beta) / (1 + beta * (mask / mask_sin) ** 2)

        return (enh[0] * pf, enh[1] * pf)


def init_dfnet3(
    sr: int = 48000,
    fft_size: int = 960,
    hop_size: int = 480,
    nb_erb: int = 32,
    nb_df: int = 96,
    df_order: int = 5,
    run_df: bool = True,
) -> DFNet3:
    """Initialize DFNet3 model with default parameters.

    Args:
        sr: Sample rate
        fft_size: FFT size
        hop_size: Hop size
        nb_erb: Number of ERB bands
        nb_df: Number of DF frequency bins
        df_order: DF filter order
        run_df: Whether to run DF stage

    Returns:
        Initialized DFNet3 model
    """
    p = ModelParams3()
    p.sr = sr
    p.fft_size = fft_size
    p.hop_size = hop_size
    p.nb_erb = nb_erb
    p.nb_df = nb_df
    p.df_order = df_order

    # Create filterbanks
    erb_fb_matrix = erb_fb(p.erb_widths, sr, inverse=False)
    erb_inv = erb_fb(p.erb_widths, sr, inverse=True)

    return DFNet3(erb_fb_matrix, erb_inv, run_df=run_df, p=p)


# Test
if __name__ == "__main__":
    print("Testing DFNet3...")

    # Create simple test
    p = ModelParams3()
    p.nb_erb = 32
    p.nb_df = 96
    p.df_order = 5
    p.fft_size = 960

    # Dummy filterbanks
    freq_bins = p.fft_size // 2 + 1
    erb_fb_matrix = mx.random.normal((p.nb_erb, freq_bins)) * 0.1
    erb_inv = mx.random.normal((freq_bins, p.nb_erb)) * 0.1

    model = DFNet3(erb_fb_matrix, erb_inv, run_df=True, p=p)

    # Test inputs
    batch, time = 2, 10
    spec_real = mx.random.normal((batch, time, freq_bins))
    spec_imag = mx.random.normal((batch, time, freq_bins))
    feat_erb = mx.random.normal((batch, time, p.nb_erb))
    feat_spec = mx.random.normal((batch, time, p.nb_df, 2))

    # Forward
    out_real, out_imag = model((spec_real, spec_imag), feat_erb, feat_spec)

    print(f"  Input spec: ({spec_real.shape}, {spec_imag.shape})")
    print(f"  Output spec: ({out_real.shape}, {out_imag.shape})")

    # Count parameters
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif isinstance(v, mx.array):
                total += v.size
        return total

    n_params = count_params(model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("✓ DFNet3 test passed!")
