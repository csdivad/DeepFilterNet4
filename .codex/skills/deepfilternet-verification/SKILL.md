---
name: deepfilternet-verification
description: Verification workflow for DeepFilterNet code, docs, and local agentic assets.
---

# DeepFilterNet Verification

Use this skill to turn claims into evidence.

## Verification layers

1. Changed-path checks:
   - Python: targeted `pytest`
   - Rust: targeted `cargo test` or `cargo build`
   - Docs/config: exact content confirmation
2. Cross-path checks:
   - related docs updated,
   - related tests considered,
   - no duplicate pathways introduced.
3. Local control-plane checks:
   - `python3 .codex/scripts/validate_agentic_setup.py`

## Reporting format

Always report:

- command run,
- pass/fail status,
- what was not verified and why.
