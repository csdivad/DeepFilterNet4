from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "build_mlx_datastore.sh"
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "download_datasets.sh"


def _write_wav(path: Path, *, sample_rate: int, seconds: float, frequency_hz: float = 440.0) -> None:
    frames = int(sample_rate * seconds)
    amplitude = 12000
    samples = bytearray()
    for i in range(frames):
        value = int(amplitude * math.sin((2.0 * math.pi * frequency_hz * i) / sample_rate))
        samples.extend(int(value).to_bytes(2, byteorder="little", signed=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_build_mlx_datastore_help_mentions_preprocess_and_merge_short() -> None:
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--preprocess-clean-speech" in result.stdout
    assert "--preprocess-probe-workers" in result.stdout
    assert "--preprocess-probe-cache" in result.stdout
    assert "DeepFilterNet3-MLX" in result.stdout
    assert "--merge-short" in result.stdout
    assert "Examples:" in result.stdout


def test_build_mlx_datastore_smoke_prints_cache_dir_override(tmp_path: Path) -> None:
    sample_rate = 16_000
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    cache_dir = tmp_path / "cache"

    speech_file = tmp_path / "speech" / "speech.wav"
    noise_file = tmp_path / "noise" / "noise.wav"
    rir_file = tmp_path / "rir" / "rir.wav"

    _write_wav(speech_file, sample_rate=sample_rate, seconds=1.2)
    _write_wav(noise_file, sample_rate=sample_rate, seconds=0.6, frequency_hz=220.0)
    _write_wav(rir_file, sample_rate=sample_rate, seconds=0.1, frequency_hz=110.0)

    lists_dir.mkdir(parents=True, exist_ok=True)
    clean_list = lists_dir / "clean_all.txt"
    noise_list = lists_dir / "noise_music.txt"
    rir_list = lists_dir / "rir_all.txt"
    clean_list.write_text(f"{speech_file}\n", encoding="utf-8")
    noise_list.write_text(f"{noise_file}\n", encoding="utf-8")
    rir_list.write_text(f"{rir_file}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--output-dir",
            str(cache_dir),
            "--clean-list",
            str(clean_list),
            "--noise-list",
            str(noise_list),
            "--rir-list",
            str(rir_list),
            "--profile",
            "prototype",
            "--sample-rate",
            str(sample_rate),
            "--segment-length",
            "1.0",
            "--min-duration",
            "0",
            "--num-workers",
            "1",
            "--shard-size",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (cache_dir / "config.json").exists(), result.stdout
    assert (cache_dir / "index.json").exists(), result.stdout
    assert "df_mlx.validate_audio_cache" in result.stdout
    assert "--cache-dir" in result.stdout
    assert str(cache_dir) in result.stdout


def test_download_datasets_help_mentions_defaults_and_cli_env_flags() -> None:
    result = subprocess.run(
        ["bash", str(DOWNLOAD_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    expected_data_dir = "/Volumes/TrainingData/datasets"
    if not Path(expected_data_dir).exists():
        expected_data_dir = str(REPO_ROOT / "data")

    assert result.returncode == 0, result.stderr
    assert f"default: {expected_data_dir}" in result.stdout
    assert "default: prototype" in result.stdout
    assert "--agree-licenses" in result.stdout
    assert "--keep-archives" in result.stdout
    assert "--verify-cache-file PATH" in result.stdout
    assert "--download-vctk / --no-download-vctk" in result.stdout
    assert "--vctk-dir PATH" in result.stdout
    assert "--librispeech-parts STRING" in result.stdout
    assert "default: 16" in result.stdout
    assert "default: 8" in result.stdout
    assert "default: none" in result.stdout


def test_download_datasets_uses_zip_merge_progress_helper() -> None:
    script_text = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")
    assert "zip_merge_progress.py" in script_text
    assert '--download-dir "${DOWNLOAD_DIR}"' in script_text
    assert '--zip-base "${zip_base}"' in script_text


def test_download_datasets_no_download_accepts_cli_overrides(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    lists_dir = tmp_path / "lists"
    downloads_dir = tmp_path / "downloads"
    extract_dir = tmp_path / "raw"

    vctk_dir = tmp_path / "existing" / "VCTK-Corpus-0.92"
    musan_dir = tmp_path / "existing" / "musan"
    air_rir_dir = tmp_path / "existing" / "air"

    _touch(vctk_dir / "wav48_silence_trimmed" / "p001" / "sample.flac")
    _touch(musan_dir / "noise" / "noise.wav")
    _touch(musan_dir / "music" / "music.wav")
    _touch(air_rir_dir / "room.wav")

    result = subprocess.run(
        [
            "bash",
            str(DOWNLOAD_SCRIPT),
            "--no-download",
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--download-dir",
            str(downloads_dir),
            "--extract-dir",
            str(extract_dir),
            "--profile",
            "production",
            "--agree-licenses",
            "--keep-archives",
            "--no-resume",
            "--no-aria2",
            "--no-aria2-parallel",
            "--aria2-conn",
            "4",
            "--aria2-split",
            "4",
            "--aria2-min-split",
            "2M",
            "--aria2-max-concurrent",
            "2",
            "--aria2-file-alloc",
            "none",
            "--aria2-user-agent",
            "TestAgent/1.0",
            "--zenodo-referer",
            "https://example.com/zenodo",
            "--no-verify-cache",
            "--verify-cache-file",
            str(tmp_path / "verify.tsv"),
            "--no-gh-auth",
            "--no-audb",
            "--install-audb",
            "--audb-dir",
            str(tmp_path / "audb"),
            "--download-vctk",
            "--no-download-librispeech",
            "--download-musan",
            "--no-download-fsd50k",
            "--download-air",
            "--no-download-openair",
            "--no-download-acousticrooms",
            "--vctk-dir",
            str(vctk_dir),
            "--librispeech-dir",
            str(tmp_path / "missing-librispeech"),
            "--musan-dir",
            str(musan_dir),
            "--fsd50k-dir",
            str(tmp_path / "missing-fsd50k"),
            "--air-rir-dir",
            str(air_rir_dir),
            "--openair-dir",
            str(tmp_path / "missing-openair"),
            "--acousticrooms-dir",
            str(tmp_path / "missing-acousticrooms"),
            "--vctk-url",
            "https://example.com/vctk.zip",
            "--librispeech-parts",
            "dev-clean test-clean",
            "--fsd50k-base-url",
            "https://example.com/fsd50k",
            "--air-version",
            "9.9.9",
            "--openair-version",
            "8.8.8",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"[config] profile=production download=0 data_dir={data_dir}" in result.stdout
    assert f"[config] download_dir={downloads_dir} extract_dir={extract_dir} list_dir={lists_dir}" in result.stdout
    assert (lists_dir / "clean_all.txt").exists(), result.stdout
    assert (lists_dir / "noise_music.txt").exists(), result.stdout
    assert (lists_dir / "rir_all.txt").exists(), result.stdout
    assert str(vctk_dir / "wav48_silence_trimmed" / "p001" / "sample.flac") in (lists_dir / "clean_all.txt").read_text()
    combined_noise = (lists_dir / "noise_music.txt").read_text()
    assert str(musan_dir / "noise" / "noise.wav") in combined_noise
    assert str(musan_dir / "music" / "music.wav") in combined_noise
    assert str(air_rir_dir / "room.wav") in (lists_dir / "rir_all.txt").read_text()


def test_download_datasets_skips_completed_processing_for_existing_archive_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    lists_dir = data_dir / "lists"
    downloads_dir = data_dir / "downloads"
    extract_dir = data_dir / "raw"

    vctk_extract_dir = extract_dir / "VCTK-Corpus-0.92"
    musan_dir = tmp_path / "existing" / "musan"
    air_rir_dir = tmp_path / "existing" / "air"

    _touch(vctk_extract_dir / "wav48_silence_trimmed" / "p001" / "sample.flac")
    _touch(vctk_extract_dir / "speaker-info.txt")
    _touch(musan_dir / "noise" / "noise.wav")
    _touch(musan_dir / "music" / "music.wav")
    _touch(air_rir_dir / "room.wav")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    (downloads_dir / "VCTK-Corpus-0.92.zip").write_bytes(b"placeholder archive")

    result = subprocess.run(
        [
            "bash",
            str(DOWNLOAD_SCRIPT),
            "--no-download",
            "--data-dir",
            str(data_dir),
            "--list-dir",
            str(lists_dir),
            "--download-dir",
            str(downloads_dir),
            "--extract-dir",
            str(extract_dir),
            "--vctk-dir",
            str(vctk_extract_dir),
            "--musan-dir",
            str(musan_dir),
            "--air-rir-dir",
            str(air_rir_dir),
            "--no-download-librispeech",
            "--no-download-fsd50k",
            "--no-download-openair",
            "--no-download-acousticrooms",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "unzip:" not in result.stderr.lower()
    assert (lists_dir / "clean_all.txt").exists(), result.stdout
    assert (
        str(vctk_extract_dir / "wav48_silence_trimmed" / "p001" / "sample.flac")
        in (lists_dir / "clean_all.txt").read_text()
    )
