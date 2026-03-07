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

    def fake_save_audio(path: str, audio, sr: int, output_dir=None, suffix=None, log=False):
        saved_targets.append(path)
        Path(path).write_bytes(b"enhanced")

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
        ),
    )
    monkeypatch.setattr(module, "init_df", lambda **kwargs: (object(), object(), None, None))
    monkeypatch.setattr(module, "ModelParams", lambda: FakeParams())
    monkeypatch.setattr(module, "AudioDataset", FakeDataset)
    monkeypatch.setattr(module, "DataLoader", FakeLoader)
    monkeypatch.setattr(module, "enhance", lambda model, df_state, audio, pad, device: audio + 1)
    monkeypatch.setattr(module, "resample", lambda audio, orig_sr, new_sr: audio)
    monkeypatch.setattr(module, "save_audio", fake_save_audio)

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
    stats.save_count = 6
    stats.save_seconds = 1.2
    stats.queue_high_water = 4

    postfix = module.build_progress_postfix(stats, inflight_saves=3, now=14.0)

    assert postfix == {
        "fps": "2.00",
        "save_q": "3",
        "enh": "300ms",
        "save": "200ms",
    }
