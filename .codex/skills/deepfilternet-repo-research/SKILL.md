---
name: deepfilternet-repo-research
description: Codanna-first repository research workflow for DeepFilterNet structure, behavior, tests, and docs.
---

# DeepFilterNet Repo Research

Use this skill before broad edits, refactors, or architectural claims.

## Workflow

1. Start with `#file:../../instructions/codebase-search-methodology.instructions.md`.
2. Map the active subsystem:
   - `DeepFilterNet/df_mlx`
   - `DeepFilterNet/df`
   - `libDF`
   - `pyDF*`
   - `ladspa`
   - `docs`
3. Identify adjacent tests and scripts.
4. Read the canonical docs for the behavior you are changing.
5. Summarize findings with file paths and why they matter.

## Fallback policy

Use `rg` only for exact literals, cleanup validation, or when Codanna cannot
answer the question. State the reason briefly.
