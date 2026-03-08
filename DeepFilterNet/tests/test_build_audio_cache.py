from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from df_mlx._audio_io import load_audio_file, resample_audio
from df_mlx.build_audio_cache import AsyncShardWriter


def _write_wav(path: Path, *, sample_rate: int, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_resample_audio_preserves_target_length_and_dtype() -> None:
    sample_rate = 44_100
    target_rate = 48_000
    duration_seconds = 0.75
    t = np.arange(int(sample_rate * duration_seconds), dtype=np.float32) / sample_rate
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    resampled = resample_audio(audio, sample_rate, target_rate)

    assert resampled.dtype == np.float32
    assert resampled.flags["C_CONTIGUOUS"]
    assert len(resampled) == int(len(audio) * target_rate / sample_rate)
    assert np.max(np.abs(resampled)) > 0.01


def test_load_audio_file_resamples_with_shared_polyphase_helper(tmp_path: Path) -> None:
    sample_rate = 16_000
    target_rate = 48_000
    duration_seconds = 0.2
    t = np.arange(int(sample_rate * duration_seconds), dtype=np.float32) / sample_rate
    audio = (0.6 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, sample_rate=sample_rate, samples=audio)

    loaded = load_audio_file(str(wav_path), target_rate)

    assert loaded.dtype == np.float32
    assert loaded.flags["C_CONTIGUOUS"]
    assert len(loaded) == int(len(audio) * target_rate / sample_rate)
    assert np.max(np.abs(loaded)) > 0.01


def test_async_shard_writer_streams_arrays_and_embeds_paths(tmp_path: Path) -> None:
    writer = AsyncShardWriter(output_dir=tmp_path, category="speech", shard_size=2)
    audio_a = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    audio_b = np.linspace(1.0, -1.0, 24, dtype=np.float32)

    writer.add("/dataset/a.wav", audio_a)
    writer.add("/dataset/b.wav", audio_b)
    index = writer.finalize()

    shard_path = tmp_path / "speech" / "shard_0000.npz"
    assert shard_path.exists()
    assert index["/dataset/a.wav"] == ("speech/shard_0000.npz", "audio_00000")
    assert index["/dataset/b.wav"] == ("speech/shard_0000.npz", "audio_00001")

    with np.load(shard_path, allow_pickle=True) as shard:
        assert list(shard["__paths__"]) == ["/dataset/a.wav", "/dataset/b.wav"]
        np.testing.assert_allclose(shard["audio_00000"], audio_a)
        np.testing.assert_allclose(shard["audio_00001"], audio_b)
