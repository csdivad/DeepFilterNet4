---
name: deepfilternet-codebase-search-methodology
description: Codanna-first repository investigation for DeepFilterNet.
applyTo: "agents/*.agent.md"
---

# DeepFilterNet Search Methodology

Use this instruction whenever a task requires repository investigation.

## Required skill

- `#skill:deepfilternet-repo-research`

## Search order

1. Check Codanna index readiness.
2. Start broad with semantic context search.
3. Narrow to symbols and symbol kinds.
4. Trace callers, callees, and impact.
5. Confirm findings with direct file reads before edits.
6. Use `rg` only for exact literals, final cleanup validation, or index
   fallback; say why.

## Canonical subsystem map

- `DeepFilterNet/df_mlx/`: active MLX training and inference path
- `DeepFilterNet/df/`: broader Python training and inference code
- `DeepFilterNet/tests/`: pytest coverage
- `libDF/`: Rust DSP/runtime core
- `pyDF/`, `pyDF-data/`, `pyDF-augment/`: Python bindings and data loaders
- `ladspa/`: LADSPA plugin
- `docs/`: design, loss, preset, and process documentation
- `scripts/`: repo automation and helper scripts

## Investigation output standard

Report:

- relevant files,
- relevant symbols,
- why each matters,
- fallback reason if you leave Codanna.
