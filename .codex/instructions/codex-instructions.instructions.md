---
name: deepfilternet-codex-instructions
description: Repo-local Codex operating contract for DeepFilterNet.
applyTo: "**"
---

# DeepFilterNet Codex Operating Contract

This instruction is the non-negotiable baseline for repo-local Codex work.

## 1) Orchestrate non-trivial work

For any task beyond a small, single-file edit:

1. Define done criteria.
2. Keep a truthful compact plan.
3. Delegate specialized work instead of doing everything serially yourself.
4. Integrate outputs and resolve conflicts.
5. Verify with evidence before handoff.

## 2) Use the local specialist map

Default routing:

- Design and cohesion: `#file:../agents/dfn-architect-strategist.agent.md`
- Python and MLX changes: `#file:../agents/dfn-python-mlx-engineer.agent.md`
- Rust and DSP changes: `#file:../agents/dfn-rust-dsp-engineer.agent.md`
- Bugs and failures: `#file:../agents/dfn-debugger-problem-solver.agent.md`
- Tests and regressions: `#file:../agents/dfn-test-engineer.agent.md`
- Docs and guides: `#file:../agents/dfn-documentation-writer.agent.md`
- Release and closeout: `#file:../agents/dfn-release-engineer.agent.md`

## 3) Shadow docs are overlays, not replacements

When working inside `DeepFilterNet/`, do not stop at `DeepFilterNet/AGENTS.md`.
Always carry forward the root `AGENTS.md` rules as the repository-wide source
of truth.

## 4) Search is Codanna-first

Follow `#file:./codebase-search-methodology.instructions.md`.

## 5) Reuse before create

- Prefer existing abstractions in `DeepFilterNet/df_mlx`, `DeepFilterNet/df`,
  `libDF`, `pyDF*`, and `ladspa`.
- Avoid shadow config systems or parallel scripts when a canonical path exists.

## 6) Evidence is mandatory

Accept work only with:

- changed files or symbols,
- test or command evidence,
- bounded risks or blockers,
- updated Beads state for non-trivial work.

## 7) Load the right depth, not all context

- Use `#skill:deepfilternet-multi-agent-orchestration` for large workstreams.
- Use `#skill:deepfilternet-repo-research` before broad changes.
- Use language/domain skills only when the task reaches that boundary.
