from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_SCRIPT = REPO_ROOT / "scripts" / "datasets" / "preprocess_clean_speech.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("preprocess_clean_speech_test_module", PREPROCESS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_resumes_existing_outputs_by_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    base_dir = tmp_path / "raw"
    output_root = tmp_path / "preprocessed"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    source_a = base_dir / "speaker_a" / "a.wav"
    source_b = base_dir / "speaker_b" / "b.wav"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    target_a = output_root / "speaker_a" / "a.wav"
    target_b = output_root / "speaker_b" / "b.wav"
    target_a.parent.mkdir(parents=True, exist_ok=True)
    target_a.write_bytes(b"already-done")

    seen_dataset_files: list[str] = []
    seen_loader_kwargs: dict[str, object] = {}
    progress_config: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, files: list[str], sr: int):
            seen_dataset_files[:] = files
            self.files = files

        def __len__(self) -> int:
            return len(self.files)

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            seen_loader_kwargs.update(kwargs)
            self.dataset = dataset

        def __iter__(self):
            for file in self.dataset.files:
                yield [file], torch.zeros(1, 8), torch.tensor([16_000])

    class FakeParams:
        sr = 16_000

    saved_targets: list[str] = []
    fake_durations = {
        source_a.resolve(): 1.25,
        source_b.resolve(): 2.75,
    }

    def fake_save_audio(path: str, audio, sr: int, output_dir=None, suffix=None, log=False):
        saved_targets.append(path)
        Path(path).write_bytes(b"enhanced")

    class FakeProgress:
        def __init__(self, iterable=None, **kwargs):
            progress_config.update(kwargs)
            self._iterable = [] if iterable is None else iterable
            self._updated = 0.0

        def __iter__(self):
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_postfix(self, value):
            progress_config["postfix"] = value

        def set_postfix_str(self, value, refresh=True):
            progress_config["postfix"] = value
            progress_config["postfix_refresh"] = refresh

        def update(self, amount):
            self._updated += amount
            progress_config["updated"] = self._updated

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=2,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )
    monkeypatch.setattr(module, "resolve_ffprobe_bin", lambda: "ffprobe")
    monkeypatch.setattr(module, "probe_audio_durations", lambda paths, ffprobe_bin: dict(fake_durations))
    monkeypatch.setattr(module, "init_df", lambda **kwargs: (object(), object(), None, None))
    monkeypatch.setattr(module, "ModelParams", lambda: FakeParams())
    monkeypatch.setattr(module, "AudioDataset", FakeDataset)
    monkeypatch.setattr(module, "DataLoader", FakeLoader)
    monkeypatch.setattr(module, "enhance", lambda model, df_state, audio, pad, device: audio + 1)
    monkeypatch.setattr(module, "resample", lambda audio, orig_sr, new_sr: audio)
    monkeypatch.setattr(module, "save_audio", fake_save_audio)
    monkeypatch.setattr(module, "tqdm", FakeProgress)

    rc = module.main()

    assert rc == 0
    assert seen_dataset_files == [str(source_b.resolve())]
    assert seen_loader_kwargs["persistent_workers"] is True
    assert seen_loader_kwargs["prefetch_factor"] == 2
    assert target_a.read_bytes() == b"already-done"
    assert target_b.exists()
    assert target_b.read_bytes() == b"enhanced"
    assert saved_targets == [str(module.build_temp_output_path(target_b))]
    output_list_lines = output_list.read_text(encoding="utf-8").splitlines()
    assert output_list_lines == [str(target_a), str(target_b)]
    assert progress_config["total"] == 4.0
    assert progress_config["initial"] == 1.25
    assert progress_config["updated"] == 2.75
    assert progress_config["bar_format"] == "{desc}: {percentage:3.0f}%|{bar}| {elapsed}{postfix}"
    postfix = str(progress_config["postfix"])
    assert "audio=4.0s/4.0s" in postfix
    assert "eta=00:00" in postfix
    assert "save=" in postfix


def test_main_fully_resumed_run_skips_ffprobe_and_backend_init(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    base_dir = tmp_path / "raw"
    output_root = tmp_path / "preprocessed"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    source_a = base_dir / "speaker_a" / "a.wav"
    source_b = base_dir / "speaker_b" / "b.wav"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    target_a = output_root / "speaker_a" / "a.wav"
    target_b = output_root / "speaker_b" / "b.wav"
    target_a.parent.mkdir(parents=True, exist_ok=True)
    target_b.parent.mkdir(parents=True, exist_ok=True)
    target_a.write_bytes(b"done-a")
    target_b.write_bytes(b"done-b")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=2,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_ffprobe_bin",
        lambda: (_ for _ in ()).throw(AssertionError("ffprobe should not run when resume is already complete")),
    )
    monkeypatch.setattr(
        module,
        "resolve_backend",
        lambda model_base_dir, requested_device: (_ for _ in ()).throw(
            AssertionError("backend init should not run when resume is already complete")
        ),
    )

    rc = module.main()

    assert rc == 0
    assert output_list.read_text(encoding="utf-8").splitlines() == [str(target_a), str(target_b)]


def test_main_rejects_colliding_output_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    source_root_a = tmp_path / "external_a"
    source_root_b = tmp_path / "external_b"
    output_root = tmp_path / "preprocessed"
    base_dir = tmp_path / "raw"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    source_a = source_root_a / "speaker_a" / "shared.wav"
    source_b = source_root_b / "speaker_b" / "shared.wav"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    file_list.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=0,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )

    try:
        module.main()
    except SystemExit as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected colliding output paths to be rejected")

    assert "Multiple source files map to the same preprocess output path" in message
    assert "shared.wav" in message


def test_resolve_backend_prefers_mlx_for_mlx_models_on_apple_silicon(monkeypatch) -> None:
    module = _load_module()
    mlx_backend = module.EnhanceBackend(name="mlx", sample_rate=48_000, enhance_audio=lambda audio: audio)

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_mlx_backend", lambda model_base_dir: mlx_backend)

    backend = module.resolve_backend("DeepFilterNet4-MLX", requested_device=None)

    assert backend is mlx_backend


def test_resolve_backend_uses_torch_for_default_deepfilternet3(monkeypatch) -> None:
    module = _load_module()
    torch_backend = module.EnhanceBackend(name="torch", sample_rate=48_000, enhance_audio=lambda audio: audio)

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_torch_backend", lambda model_base_dir, requested_device: torch_backend)

    backend = module.resolve_backend("DeepFilterNet3", requested_device=None)

    assert backend is torch_backend


def test_resolve_backend_prefers_mlx_for_custom_model_dirs_on_apple_silicon(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    mlx_backend = module.EnhanceBackend(name="mlx", sample_rate=48_000, enhance_audio=lambda audio: audio)
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()

    monkeypatch.setattr(module, "running_on_apple_silicon", lambda: True)
    monkeypatch.setattr(module, "load_mlx_backend", lambda model_base_dir: mlx_backend)

    backend = module.resolve_backend(str(model_dir), requested_device=None)

    assert backend is mlx_backend


def test_main_rejects_obvious_non_speech_sources_by_default(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    base_dir = tmp_path / "raw"
    output_root = tmp_path / "preprocessed"
    file_list = tmp_path / "clean.txt"
    output_list = tmp_path / "clean.preprocessed.txt"

    noise_source = base_dir / "musan" / "noise" / "noise.wav"
    noise_source.parent.mkdir(parents=True, exist_ok=True)
    noise_source.write_bytes(b"noise")
    file_list.write_text(f"{noise_source}\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            file_list=str(file_list),
            output_root=str(output_root),
            base_dir=str(base_dir),
            output_list=str(output_list),
            model_base_dir="DeepFilterNet3",
            device=None,
            num_workers=0,
            overwrite=False,
            allow_non_speech_paths=False,
        ),
    )
    monkeypatch.setattr(module, "resolve_ffprobe_bin", lambda: "ffprobe")
    monkeypatch.setattr(module, "probe_audio_durations", lambda paths, ffprobe_bin: {noise_source.resolve(): 0.5})

    try:
        module.main()
    except SystemExit as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected obvious non-speech input to be rejected")

    assert "Refusing to preprocess obvious non-speech inputs" in message
    assert str(noise_source.resolve()) in message


def test_save_enhanced_audio_atomically_cleans_partial_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    target = tmp_path / "out" / "sample.wav"
    temp_target = module.build_temp_output_path(target)

    def failing_save_audio(path: str, audio, sr: int, output_dir=None, suffix=None, log=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"partial")
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "save_audio", failing_save_audio)
    monkeypatch.setattr(module, "resample", lambda audio, orig_sr, new_sr: audio)

    try:
        module.save_enhanced_audio_atomically(target, torch.zeros(1, 4), df_sr=16_000, orig_sr=16_000)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected save_enhanced_audio_atomically to raise")

    assert not temp_target.exists()
    assert not target.exists()


def test_build_progress_postfix_reports_rate_queue_and_stage_timings() -> None:
    module = _load_module()
    stats = module.PreprocessProgressStats(start_time=10.0)
    stats.enhance_count = 8
    stats.enhance_seconds = 2.4
    stats.processed_audio_seconds = 12.0
    stats.save_count = 6
    stats.save_seconds = 1.2
    stats.queue_high_water = 4

    postfix = module.build_progress_postfix(
        stats,
        inflight_saves=3,
        completed_audio_seconds=18.0,
        total_audio_seconds=30.0,
        now=14.0,
    )

    assert postfix == "audio=18.0s/30.0s, eta=00:04, rt=3.00x, save_q=3, enh=300ms, save=200ms"
