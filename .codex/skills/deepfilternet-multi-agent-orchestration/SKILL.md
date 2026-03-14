---
name: deepfilternet-multi-agent-orchestration
description: DeepFilterNet-specific multi-agent planning, delegation, integration, and verification workflow.
---

# DeepFilterNet Multi-Agent Orchestration

Use this skill for any task that crosses subsystems, languages, or stages.

## Core behavior

1. Load `#file:../../instructions/codex-instructions.instructions.md`.
2. Define done criteria first.
3. Create a compact plan with clear ownership.
4. Delegate architecture, implementation, debugging, tests, docs, and release
   work to the matching local specialist agents.
5. Integrate outputs and resolve conflicts yourself.
6. Run a verification pass before handoff.

## Delegation guide

- Architecture and duplicate-path decisions:
  `#file:../../agents/dfn-architect-strategist.agent.md`
- Python and MLX implementation:
  `#file:../../agents/dfn-python-mlx-engineer.agent.md`
- Rust and DSP implementation:
  `#file:../../agents/dfn-rust-dsp-engineer.agent.md`
- Failure investigation:
  `#file:../../agents/dfn-debugger-problem-solver.agent.md`
- Test design and regression checks:
  `#file:../../agents/dfn-test-engineer.agent.md`
- Documentation updates:
  `#file:../../agents/dfn-documentation-writer.agent.md`
- Final closeout:
  `#file:../../agents/dfn-release-engineer.agent.md`

## Handoff contract

Each delegated task must include:

- scope,
- boundaries,
- files or subsystems involved,
- evidence required,
- explicit done criteria.
