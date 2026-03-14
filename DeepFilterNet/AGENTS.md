# DeepFilterNet Subtree Instructions

This file is an overlay for work inside `DeepFilterNet/`. The authoritative
repository-wide contract still lives at the repo root in `../AGENTS.md`, and
the repo-local Codex entrypoint lives at `../.codex/AGENTS.md`. Do not treat
this file as a replacement for either one.

## Local focus

- The most actively developed path is `df_mlx/`.
- Python package commands should be run from this directory unless the task is
  explicitly workspace-wide.
- Keep `README.md`, `df_mlx/README.md`, and the relevant `docs/` pages aligned
  when changing runtime or config semantics.

## Quick Reference

```bash
bd prime
bd ready
python -m pytest
python -m df_mlx.train_dynamic --print-run-config
```

## Non-negotiables carried from the root contract

- Use Beads for non-trivial work.
- Push before handoff.
- Preserve repository identity as `sealad886/DeepFilterNet4`.
- Prefer Codanna-first investigation and evidence-backed verification.
