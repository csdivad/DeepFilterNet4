"""Regression guard for train/valid dataset split isolation.

Background:
A runtime crash was observed when MLX prefetch workers loaded train indices
while the shared dataset had been switched to the ``valid`` split:
"IndexError: Sample index ... out of range for split 'valid'".

To prevent this class of bug, training should keep a dedicated validation
DynamicDataset instance (separate mutable split/epoch state) and wire
ValidationContext to that instance.
"""

from __future__ import annotations

from pathlib import Path

_TRAIN_DYNAMIC_PATH = Path(__file__).resolve().parent.parent / "df_mlx" / "train_dynamic.py"


def test_train_uses_dedicated_validation_dataset_instance() -> None:
    source = _TRAIN_DYNAMIC_PATH.read_text(encoding="utf-8")
    assert "valid_dataset = DynamicDataset(copy.deepcopy(config))" in source


def test_validation_context_uses_validation_dataset() -> None:
    source = _TRAIN_DYNAMIC_PATH.read_text(encoding="utf-8")
    ctx_pos = source.find("ValidationContext(")
    assert ctx_pos != -1, "ValidationContext construction not found"
    dataset_arg_pos = source.find("dataset=valid_dataset", ctx_pos)
    assert dataset_arg_pos != -1, "ValidationContext must use the isolated validation dataset"


def test_curriculum_syncs_hardness_probs_to_validation_dataset() -> None:
    source = _TRAIN_DYNAMIC_PATH.read_text(encoding="utf-8")
    assert "valid_dataset.config.p_extreme_snr = cur_p_extreme" in source
    assert "valid_dataset.config.p_very_low_snr = cur_p_very_low" in source
    assert "valid_dataset.config.p_interfer_speech = cur_p_interfer" in source
