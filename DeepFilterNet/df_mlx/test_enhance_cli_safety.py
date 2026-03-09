import types

import pytest

from df_mlx import enhance as enhance_mod
from df_mlx.deepfilternet3 import Conv2dNormAct, DFNet3, ModelParams3, _apply_conv_lookahead


def test_normalize_epoch_spec_accepts_integer_string():
    assert enhance_mod.normalize_epoch_spec("3") == 3
    assert enhance_mod.normalize_epoch_spec("latest") == "latest"
    assert enhance_mod.normalize_epoch_spec("none") == "none"
    with pytest.raises(ValueError):
        enhance_mod.normalize_epoch_spec("not-an-epoch")


def test_load_model_reads_df_section_from_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.ini").write_text(
        "[df]\n" "sr = 16000\n" "fft_size = 512\n" "hop_size = 128\n" "nb_df = 48\n" "nb_erb = 24\n"
    )

    _, params, _, loaded_epoch = enhance_mod.load_model(str(model_dir), epoch="none")
    assert loaded_epoch == 0
    assert params.sr == 16000
    assert params.fft_size == 512
    assert params.hop_size == 128
    assert params.nb_df == 48
    assert params.nb_erb == 24


def test_load_model_detects_dfnet3_bundle_from_train_model(tmp_path):
    model_dir = tmp_path / "dfnet3_model"
    model_dir.mkdir()
    (model_dir / "config.ini").write_text(
        "[train]\n"
        "model = deepfilternet3\n"
        "[df]\n"
        "sr = 16000\n"
        "fft_size = 512\n"
        "hop_size = 128\n"
        "nb_df = 48\n"
        "nb_erb = 24\n"
        "min_nb_erb_freqs = 2\n"
        "[deepfilternet]\n"
        "conv_ch = 16\n",
        encoding="utf-8",
    )

    model, params, _, loaded_epoch = enhance_mod.load_model(str(model_dir), epoch="none")

    assert loaded_epoch == 0
    assert isinstance(model, DFNet3)
    assert isinstance(params, ModelParams3)
    assert params.sr == 16000
    assert params.fft_size == 512
    assert params.hop_size == 128
    assert params.nb_df == 48
    assert params.nb_erb == 24


def test_enhance_dispatches_dfnet3_to_specialized_path(monkeypatch):
    model = DFNet3.__new__(DFNet3)
    params = ModelParams3()
    sentinel = object()

    def fake_enhance_dfnet3(model_arg, audio_arg, params_arg, compensate_delay, atten_lim_db, df_state=None):
        assert model_arg is model
        assert params_arg is params
        assert compensate_delay is False
        assert atten_lim_db == 6.0
        return sentinel

    monkeypatch.setattr(enhance_mod, "_enhance_dfnet3", fake_enhance_dfnet3)

    result = enhance_mod.enhance(model, enhance_mod.mx.zeros((16,)), params, compensate_delay=False, atten_lim_db=6.0)

    assert result is sentinel


def test_apply_conv_lookahead_shifts_features_and_zero_pads_tail():
    features = enhance_mod.mx.array([[[1.0], [2.0], [3.0], [4.0]]])

    shifted = _apply_conv_lookahead(features, 2)

    assert shifted.tolist() == [[[3.0], [4.0], [0.0], [0.0]]]


def test_dfnet3_conv_wrapper_matches_torch_style_causal_time_padding():
    layer = Conv2dNormAct(
        1,
        1,
        kernel_size=(3, 1),
        bias=False,
        norm=None,
        activation=None,
        separable=False,
    )
    layer.conv.weight = enhance_mod.mx.ones_like(layer.conv.weight)

    x = enhance_mod.mx.array([[[[1.0]], [[2.0]], [[4.0]], [[8.0]]]])

    y = layer(x)

    assert y[..., 0].tolist() == [[[1.0], [3.0], [7.0], [14.0]]]


def test_main_dispatches_to_streaming_path(monkeypatch):
    calls = []

    class Params:
        sr = 48000
        hop_size = 480

    def fake_load_model(*, model_path, epoch):
        return object(), Params(), "suffix", 0

    def fake_batch(*args, **kwargs):
        calls.append("batch")
        return ["out.wav"]

    def fake_stream_batch(*args, **kwargs):
        calls.append("stream")
        return ["out.wav"]

    monkeypatch.setattr(enhance_mod, "load_model", fake_load_model)
    monkeypatch.setattr(enhance_mod, "enhance_batch", fake_batch)
    monkeypatch.setattr(enhance_mod, "enhance_batch_streaming", fake_stream_batch)

    args = types.SimpleNamespace(
        log_level="ERROR",
        input_dir=None,
        input_files=["in.wav"],
        model=None,
        epoch="best",
        suffix=None,
        output_dir=".",
        no_delay_compensation=False,
        atten_lim=None,
        streaming=True,
        streaming_chunk_ms=100.0,
        speech_boost_db=0.0,
        speech_boost_threshold=0.5,
        speech_boost_min_speech_ms=250,
        speech_boost_min_silence_ms=100,
        speech_boost_pad_ms=30,
        speech_boost_ramp_ms=10.0,
        speech_boost_peak_limit=0.99,
        speech_boost_silero_model_path=None,
        speech_boost_silero_sample_rate=16000,
    )
    rc = enhance_mod.main(args)
    assert rc == 0
    assert calls == ["stream"]


def test_main_dispatches_to_batch_path(monkeypatch):
    calls = []

    class Params:
        sr = 48000
        hop_size = 480

    def fake_load_model(*, model_path, epoch):
        return object(), Params(), "suffix", 0

    def fake_batch(*args, **kwargs):
        calls.append("batch")
        return ["out.wav"]

    def fake_stream_batch(*args, **kwargs):
        calls.append("stream")
        return ["out.wav"]

    monkeypatch.setattr(enhance_mod, "load_model", fake_load_model)
    monkeypatch.setattr(enhance_mod, "enhance_batch", fake_batch)
    monkeypatch.setattr(enhance_mod, "enhance_batch_streaming", fake_stream_batch)

    args = types.SimpleNamespace(
        log_level="ERROR",
        input_dir=None,
        input_files=["in.wav"],
        model=None,
        epoch="best",
        suffix=None,
        output_dir=".",
        no_delay_compensation=False,
        atten_lim=None,
        streaming=False,
        streaming_chunk_ms=100.0,
        speech_boost_db=0.0,
        speech_boost_threshold=0.5,
        speech_boost_min_speech_ms=250,
        speech_boost_min_silence_ms=100,
        speech_boost_pad_ms=30,
        speech_boost_ramp_ms=10.0,
        speech_boost_peak_limit=0.99,
        speech_boost_silero_model_path=None,
        speech_boost_silero_sample_rate=16000,
    )
    rc = enhance_mod.main(args)
    assert rc == 0
    assert calls == ["batch"]
