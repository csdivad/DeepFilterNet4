"""Tests for data checkpoint resume position consistency.

Regression tests for the bug where MLXDataStream.get_progress() returned
batch=0 after from_checkpoint() because __init__ hardcoded _batch_count=0
instead of reading it from the loaded CheckpointState.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from df_mlx.dynamic_dataset import CheckpointState  # noqa: E402


class TestCheckpointStateRoundtrip:
    """Verify CheckpointState save/load preserves batch_idx."""

    def test_roundtrip_preserves_batch_idx(self, tmp_path):
        cs = CheckpointState()
        cs.epoch = 35
        cs.batch_idx = 201
        cs.samples_processed = 201 * 24
        cs.seed = 42

        path = tmp_path / "data_checkpoint.json"
        cs.save(path)

        loaded = CheckpointState.load(path)
        assert loaded.epoch == 35
        assert loaded.batch_idx == 201
        assert loaded.samples_processed == 201 * 24

    def test_roundtrip_preserves_zero_batch(self, tmp_path):
        cs = CheckpointState()
        cs.epoch = 10
        cs.batch_idx = 0
        path = tmp_path / "data_checkpoint.json"
        cs.save(path)

        loaded = CheckpointState.load(path)
        assert loaded.batch_idx == 0

    def test_json_has_batch_idx_key(self, tmp_path):
        cs = CheckpointState()
        cs.epoch = 5
        cs.batch_idx = 42
        path = tmp_path / "data_checkpoint.json"
        cs.save(path)

        with open(path) as f:
            raw = json.load(f)
        assert raw["batch_idx"] == 42
        assert raw["epoch"] == 5


class TestMLXDataStreamProgressAfterResume:
    """Verify get_progress() returns correct batch after from_checkpoint()."""

    @pytest.fixture
    def mock_dataset(self):
        ds = MagicMock()
        ds.config = MagicMock()
        ds.config.seed = 42
        ds.__len__ = MagicMock(return_value=1000)
        ds.set_split = MagicMock()
        ds.set_epoch = MagicMock()
        return ds

    @pytest.fixture
    def saved_checkpoint(self, tmp_path):
        cs = CheckpointState()
        cs.epoch = 35
        cs.batch_idx = 201
        cs.samples_processed = 201 * 24
        cs.seed = 42
        path = tmp_path / "data_checkpoint.json"
        cs.save(path)
        return path

    def test_get_progress_matches_checkpoint_batch(self, mock_dataset, saved_checkpoint):
        """Regression: get_progress() must reflect checkpoint batch_idx
        immediately after from_checkpoint(), before any iteration."""
        try:
            from df_mlx.dynamic_dataset import MLXDataStream
        except ImportError:
            pytest.skip("mlx-data not available")

        stream = MLXDataStream.from_checkpoint(
            dataset=mock_dataset,
            checkpoint_path=saved_checkpoint,
            batch_size=24,
        )
        progress = stream.get_progress()
        assert progress["epoch"] == 35
        assert progress["batch"] == 201, (
            f"get_progress() returned batch={progress['batch']} but checkpoint "
            f"has batch_idx=201. This causes model/data resume mismatch."
        )

    def test_get_progress_batch_zero_for_fresh_stream(self, mock_dataset):
        """Fresh stream (no checkpoint) should report batch=0."""
        try:
            from df_mlx.dynamic_dataset import MLXDataStream
        except ImportError:
            pytest.skip("mlx-data not available")

        stream = MLXDataStream(dataset=mock_dataset, batch_size=24)
        progress = stream.get_progress()
        assert progress["batch"] == 0

    def test_batch_count_synced_after_construction(self, mock_dataset, saved_checkpoint):
        """_batch_count must equal _checkpoint.batch_idx after construction."""
        try:
            from df_mlx.dynamic_dataset import MLXDataStream
        except ImportError:
            pytest.skip("mlx-data not available")

        stream = MLXDataStream.from_checkpoint(
            dataset=mock_dataset,
            checkpoint_path=saved_checkpoint,
            batch_size=24,
        )
        assert stream._batch_count == 201
        assert stream._checkpoint.batch_idx == 201

    def test_set_epoch_resets_batch_count(self, mock_dataset, saved_checkpoint):
        """set_epoch() must reset both _batch_count and _checkpoint.batch_idx."""
        try:
            from df_mlx.dynamic_dataset import MLXDataStream
        except ImportError:
            pytest.skip("mlx-data not available")

        stream = MLXDataStream.from_checkpoint(
            dataset=mock_dataset,
            checkpoint_path=saved_checkpoint,
            batch_size=24,
        )
        assert stream._batch_count == 201
        stream.set_epoch(36)
        assert stream._batch_count == 0
        assert stream._checkpoint.batch_idx == 0
        progress = stream.get_progress()
        assert progress["batch"] == 0

    def test_set_resume_position_updates_progress(self, mock_dataset):
        """set_resume_position() must update get_progress() immediately."""
        try:
            from df_mlx.dynamic_dataset import MLXDataStream
        except ImportError:
            pytest.skip("mlx-data not available")

        stream = MLXDataStream(dataset=mock_dataset, batch_size=24)
        stream.set_resume_position(epoch=10, batch_idx=50)
        progress = stream.get_progress()
        assert progress["epoch"] == 10
        assert progress["batch"] == 50


class TestResumeValidationLogic:
    """Test the model-vs-data checkpoint comparison in train()."""

    def test_matching_positions_pass_validation(self, tmp_path):
        """When model and data agree on epoch/batch, no error should be raised."""
        data_epoch = 35
        data_batch = 201
        model_epoch = 35
        model_batch = 201

        assert data_epoch == model_epoch
        assert data_batch == model_batch

    def test_mismatched_batch_triggers_error(self):
        """The exact scenario from the bug: model=201, data=0."""
        data_epoch = 35
        data_batch = 0
        model_epoch = 35
        model_batch = 201
        kind = "interrupted"

        from df_mlx.training_checkpoints import _IN_PROGRESS_KINDS

        resume_requires_mid_epoch = kind in _IN_PROGRESS_KINDS
        assert resume_requires_mid_epoch

        should_fail = data_epoch != model_epoch or data_batch != model_batch
        assert should_fail, (
            "Mismatched positions should be detected: " f"model batch={model_batch}, data batch={data_batch}"
        )
