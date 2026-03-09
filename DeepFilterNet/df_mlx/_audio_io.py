"""Shared audio I/O helpers for data-preparation scripts.

Provides a single ``load_audio_file`` implementation used by
``dynamic_dataset``, ``build_audio_cache``, and ``prepare_data``
so that the load→mono→resample→float32 pipeline is defined in one place.

The function gracefully falls back from *soundfile* to *scipy.io.wavfile*
so that scripts work in minimal dependency environments.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio to ``target_sr`` with a polyphase filter.

    The datastore/preparation pipeline is primarily CPU-bound on Apple Silicon,
    so prefer ``resample_poly`` over FFT-based ``resample`` to reduce both
    latency and temporary allocation pressure for common audio-rate conversions.
    The output length preserves the historical ``int(len * target / orig)``
    contract used by the previous FFT-based implementation.
    """
    if orig_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)

    target_samples = max(1, int(len(audio) * target_sr / orig_sr))
    rate_gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // rate_gcd
    down = orig_sr // rate_gcd
    resampled = np.asarray(scipy_signal.resample_poly(audio, up, down), dtype=np.float32)

    if len(resampled) > target_samples:
        resampled = resampled[:target_samples]
    elif len(resampled) < target_samples:
        resampled = np.pad(resampled, (0, target_samples - len(resampled)))

    return np.ascontiguousarray(resampled, dtype=np.float32)


# ---------------------------------------------------------------------------
# Primary implementation: soundfile (preferred)
# ---------------------------------------------------------------------------
try:
    import soundfile as sf

    def load_audio_file(path: str, sr: int) -> np.ndarray:
        """Load an audio file, convert to mono float32, and resample to *sr*."""
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise ValueError(f"Audio file contains zero samples: {path}")
        if file_sr != sr:
            audio = resample_audio(audio, file_sr, sr)
        return np.ascontiguousarray(audio, dtype=np.float32)

except ImportError:
    from scipy.io import wavfile

    def load_audio_file(path: str, sr: int) -> np.ndarray:  # type: ignore[misc]
        """Load an audio file, convert to mono float32, and resample to *sr*."""
        file_sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        elif audio.dtype == np.float64:
            audio = audio.astype(np.float32)
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise ValueError(f"Audio file contains zero samples: {path}")
        if file_sr != sr:
            audio = resample_audio(audio, file_sr, sr)
        return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio_file_safe(path: str, sr: int) -> Optional[np.ndarray]:
    """Like :func:`load_audio_file` but returns ``None`` on failure."""
    try:
        return load_audio_file(path, sr)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None
