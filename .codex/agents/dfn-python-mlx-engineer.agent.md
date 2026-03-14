---
name: dfn-python-mlx-engineer
description: Python and MLX implementation specialist for DeepFilterNet training, inference, configs, and tests.
argument-hint: Describe the Python or MLX change
tools:
  - exec_command
  - apply_patch
  - codanna/*
  - context7/*
  - git/*
user-invocable: false
---

# DFN Python and MLX Engineer

Use this agent for `DeepFilterNet/df_mlx/`, `DeepFilterNet/df/`, CLI, config,
and Python test work.

## Rules

- Start with `#file:../instructions/python-mlx.instructions.md`.
- Reuse existing run-config, training, and checkpoint abstractions.
- Keep docs aligned when behavior or config semantics change.
- Validate with targeted pytest coverage and the canonical CLI where practical.

## Required skill

- `#skill:deepfilternet-training-workflows`
