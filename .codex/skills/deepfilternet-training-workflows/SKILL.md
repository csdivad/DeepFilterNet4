---
name: deepfilternet-training-workflows
description: Training, inference, run-config, and MLX workflow guidance for DeepFilterNet.
---

# DeepFilterNet Training Workflows

Use this skill for `df_mlx` training, inference, presets, checkpointing, or
loss-stack changes.

## Required references

- `#file:../../instructions/python-mlx.instructions.md`
- `README.md`
- `DeepFilterNet/df_mlx/README.md`
- `docs/RUN_CONFIG_PRESETS.md`
- `docs/LOSSES.md`

## Workflow

1. Confirm whether the task is training, inference, config translation, or
   checkpointing.
2. Read the current module and nearby tests.
3. Validate config precedence and current CLI behavior before editing.
4. Update docs when semantics change.
5. Run targeted pytest or command validation that matches the touched path.

## Non-negotiables

- Do not invent parallel config formats.
- Keep `RunConfig` and CLI docs aligned.
- Treat VAD, loss weights, and checkpoint behavior as high-risk areas needing
  explicit verification.
