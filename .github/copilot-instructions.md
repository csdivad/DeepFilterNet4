---
name: DeepFilterNet4 Repository Instructions
description: Custom GitHub Copilot instructions for the DeepFilterNet4 repository.
applyTo: "**"
---
# GitHub Copilot Workspace Instructions

## Repository Identity

- **This is sealad886/DeepFilterNet** — a standalone fork
- **There is NO upstream repository relationship**
- **NEVER create PRs to or reference Rikorose/DeepFilterNet**
- All work stays within this repository only

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context, or install hooks (`bd hooks install`) for auto-injection.

**Quick reference:**
- `bd ready` - Find unblocked work
- `bd create "Title" --type task --priority 2` - Create issue
- `bd close <id>` - Complete work
- `bd dolt push` - Push beads to remote

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
