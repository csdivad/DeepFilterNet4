---
name: DeepFilterNet4 Repository Instructions
description: Custom GitHub Copilot instructions for the DeepFilterNet4 repository.
applyTo: "**"
---
# GitHub Copilot Workspace Instructions

## Repository Identity

- **This is sealad886/DeepFilterNet4** — a standalone fork
- **There is NO upstream repository relationship**
- **NEVER create PRs to or reference Rikorose/DeepFilterNet**
- All work stays within this repository only

## Active Path

- The most actively developed code path is `DeepFilterNet/df_mlx/`.

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context. This repo keeps the Beads hook shims in
`.githooks/`, so enable them with `./setup.sh` or `scripts/install-hooks.sh`.
Use `bd hooks install` only if you intentionally want upstream-managed hooks.

**Quick reference:**
- `bd ready` - Find unblocked work
- `bd create "Title" --type task --priority 2` - Create issue
- `bd close <id>` - Complete work
- `bd dolt remote list` - Check whether a Dolt remote is configured
- `bd dolt push` - Push beads to remote **when a Dolt remote is configured**

For full workflow details: `bd prime`

## Project Structure

- `DeepFilterNet/` - Main Python package (training, inference, configs)
- `DeepFilterNet/df/` - Core Python code
- `DeepFilterNet/tests/` - Python tests (pytest)
- `libDF/`, `ladspa/` - Rust crates for DSP/runtime and LADSPA plugin
- `models/` - Pretrained model archives
- `pyDF/`, `pyDF-data/` - Python bindings and data utilities

## Build & Test

- Python: `poetry -C DeepFilterNet install`, `pytest` (from `DeepFilterNet/`)
- Rust: `cargo build`, `cargo test`

## Coding Style

- Python: Black (`line-length = 120`), isort, Pyright for type checking
- Rust: `cargo fmt` (follows `rustfmt.toml`)
- Tests: pytest, name test files `test_*.py`

## Commit Guidelines

- Follow Conventional Commits style (e.g., `feat(whisper): ...`, `chore(lint): ...`)
- Always push changes before ending a session
