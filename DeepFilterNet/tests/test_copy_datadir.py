from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COPY_DATADIR_PATH = REPO_ROOT / "scripts" / "copy_datadir.py"


def _load_copy_datadir_module():
    spec = importlib.util.spec_from_file_location("copy_datadir_test_module", COPY_DATADIR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_datasets_uses_progress_bar_without_copy_spam(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_copy_datadir_module()

    src_dir = Path("/copy-progress-src")
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    file_a = src_dir / "FSD50K.dev_audio.noise_a.hdf5"
    file_b = src_dir / "FSD50K.eval_audio.noise_b.hdf5"

    cfg_path = tmp_path / "datasets.json"
    cfg_path.write_text(
        json.dumps(
            {
                "train": [[file_a.name, 1.0], [file_b.name, 0.5]],
                "valid": [],
                "test": [],
            }
        ),
        encoding="utf-8",
    )

    class FakeH5:
        def __init__(self, path: str):
            self.path = path

        def keys(self):
            return ["noise"]

    class FakeFuture:
        def __init__(self, result=None):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, total: int, desc: str, unit: str, dynamic_ncols: bool):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.updated = 0
            self.postfix: str | None = None
            FakeTqdm.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1) -> None:
            self.updated += n

        def set_postfix_str(self, value: str) -> None:
            self.postfix = value

    copied: list[tuple[str, str, bool]] = []

    def fake_cp(src: str, tgt: str, try_other_hosts: bool = False, verbose: int = 0):
        copied.append((src, tgt, try_other_hosts))
        Path(tgt).parent.mkdir(parents=True, exist_ok=True)
        Path(tgt).write_bytes(b"copied")

    def fake_du(path: str, block_size: str = "1"):
        path_obj = Path(path)
        if path_obj == target_dir:
            return len(copied) * 1024
        return 1024

    monkeypatch.setattr(module, "cp", fake_cp)
    monkeypatch.setattr(module, "du", fake_du)
    monkeypatch.setattr(module, "has_locks", lambda *args, **kwargs: (False, False))
    monkeypatch.setattr(module, "h5py", type("FakeH5Module", (), {"File": FakeH5}))
    monkeypatch.setattr(module.concurrent.futures, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(module.concurrent.futures, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(module, "tqdm", FakeTqdm)

    module.copy_datasets(str(src_dir), str(target_dir), str(cfg_path), max_gb=1.0, lock=None, try_other_hosts=False)

    captured = capsys.readouterr()
    assert f"copying {file_a}" not in captured.out.lower()
    assert f"copying {file_b}" not in captured.out.lower()
    assert len(copied) == 2
    assert FakeTqdm.instances, "Expected tqdm progress bar to be created"
    progress = FakeTqdm.instances[0]
    assert progress.desc == "Copying datasets"
    assert progress.total == 2
    assert progress.updated == 2
    assert progress.postfix is not None and progress.postfix.endswith("GB")
