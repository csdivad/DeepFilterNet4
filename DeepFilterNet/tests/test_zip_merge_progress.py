from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "scripts" / "datasets" / "zip_merge_progress.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("zip_merge_progress_test_module", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_split_zip_with_progress_uses_tqdm_and_suppresses_copy_spam(monkeypatch, capsys, tmp_path: Path) -> None:
    module = _load_helper_module()
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    temp_output = download_dir / "FSD50K.dev_audio.zip.merged.tmp.zip"

    class FakeRunResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = "clip1.wav\nclip2.wav\n"
            self.stderr = ""

    class FakePopen:
        def __init__(self, *args, **kwargs):
            temp_output.write_bytes(b"temp zip bytes")
            self.stdout = iter(
                [
                    "copying: clip1.wav\n",
                    "copying: clip2.wav\n",
                ]
            )

        def wait(self):
            return 0

    class FakeTqdm:
        instances: list["FakeTqdm"] = []
        writes: list[str] = []

        def __init__(self, total: int, desc: str, unit: str, dynamic_ncols: bool):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.n = 0
            self.postfix: str | None = None
            FakeTqdm.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1) -> None:
            self.n += n

        def set_postfix_str(self, value: str) -> None:
            self.postfix = value

        @staticmethod
        def write(value: str) -> None:
            FakeTqdm.writes.append(value)

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: FakeRunResult())
    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(module, "tqdm", FakeTqdm)

    merged_path = module.merge_split_zip_with_progress(download_dir, "FSD50K.dev_audio.zip")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert merged_path == download_dir / "FSD50K.dev_audio.zip.merged.zip"
    assert FakeTqdm.instances, "Expected a tqdm progress bar instance"
    progress = FakeTqdm.instances[0]
    assert progress.total == 2
    assert progress.desc == "Merging FSD50K.dev_audio.zip"
    assert progress.n == 2
    assert progress.postfix == "clip2.wav"
    assert FakeTqdm.writes == []


def test_merge_split_zip_with_progress_skips_valid_existing_output(monkeypatch, capsys, tmp_path: Path) -> None:
    module = _load_helper_module()
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    merged_zip = download_dir / "FSD50K.dev_audio.zip.merged.zip"
    merged_zip.write_bytes(b"already merged")

    class FakeTqdm:
        writes: list[str] = []

        @staticmethod
        def write(value: str) -> None:
            FakeTqdm.writes.append(value)

    monkeypatch.setattr(module, "count_zip_members", lambda *args, **kwargs: 2)
    monkeypatch.setattr(module, "verify_zip_archive", lambda path: path == merged_zip)
    monkeypatch.setattr(module, "tqdm", FakeTqdm)

    merged_path = module.merge_split_zip_with_progress(download_dir, "FSD50K.dev_audio.zip")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert merged_path == merged_zip
    assert FakeTqdm.writes == [f"[skip] merged archive already exists: {merged_zip}"]


def test_merge_split_zip_with_progress_renames_temp_output_atomically(monkeypatch, tmp_path: Path) -> None:
    module = _load_helper_module()
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    temp_output = download_dir / "FSD50K.dev_audio.zip.merged.tmp.zip"
    final_output = download_dir / "FSD50K.dev_audio.zip.merged.zip"

    class FakeRunResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = "clip1.wav\n"
            self.stderr = ""

    class FakePopen:
        def __init__(self, args, cwd, stdout, stderr, text, bufsize):
            self.stdout = iter(["copying: clip1.wav\n"])
            temp_output.write_bytes(b"temp zip bytes")

        def wait(self):
            return 0

    class FakeTqdm:
        def __init__(self, total: int, desc: str, unit: str, dynamic_ncols: bool):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1) -> None:
            self.n += n

        def set_postfix_str(self, value: str) -> None:
            return None

        @staticmethod
        def write(value: str) -> None:
            return None

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: FakeRunResult())
    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(module, "tqdm", FakeTqdm)

    merged_path = module.merge_split_zip_with_progress(download_dir, "FSD50K.dev_audio.zip")

    assert merged_path == final_output
    assert final_output.exists()
    assert final_output.read_bytes() == b"temp zip bytes"
    assert not temp_output.exists()
