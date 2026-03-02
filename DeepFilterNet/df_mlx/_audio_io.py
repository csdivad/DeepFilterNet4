"""Shared audio I/O helpers for data-preparation scripts.

Provides a single ``load_audio_file`` implementation used by
``dynamic_dataset``, ``build_audio_cache``, and ``prepare_data``
so that the load→mono→resample→float32 pipeline is defined in one place.

The function gracefully falls back from *soundfile* to *scipy.io.wavfile*
so that scripts work in minimal dependency environments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

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
        if file_sr != sr:
            num_samples = int(len(audio) * sr / file_sr)
            audio = np.asarray(scipy_signal.resample(audio, num_samples), dtype=np.float32)
        return audio.astype(np.float32)

except ImportError:
    from scipy.io import wavfile

    def load_audio_file(path: str, sr: int) -> np.ndarray:  # type: ignore[misc]
        """Load an audio file, convert to mono float32, and resample to *sr*."""
        file_sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            num_samples = int(len(audio) * sr / file_sr)
            audio = np.asarray(scipy_signal.resample(audio, num_samples), dtype=np.float32)
        return audio.astype(np.float32)


def load_audio_file_safe(path: str, sr: int) -> Optional[np.ndarray]:
    """Like :func:`load_audio_file` but returns ``None`` on failure."""
    try:
        return load_audio_file(path, sr)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None
