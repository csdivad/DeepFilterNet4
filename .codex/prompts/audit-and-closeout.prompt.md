---
name: DeepFilterNet Audit and Closeout
description: Run a DeepFilterNet-specific audit, remediate findings, verify with evidence, and finish repo closeout.
agent: dfn-shadow-orchestrator
---

Run a rigorous audit and closeout pass for this repository.

## Workflow

1. Initialize Beads control:
   - run `bd prime`
   - claim or create the issue
   - keep an explicit plan
2. Investigate with Codanna first.
3. Review changed scope from three angles:
   - correctness and regressions
   - doc and config alignment
   - release readiness and closeout gaps
4. Fix in-scope findings.
5. Run targeted checks plus any broad regression gates that fit the change.
6. If `.codex` assets changed, run:
   - `python3 .codex/scripts/validate_agentic_setup.py`
7. Finish Beads, commit, push, and verify final git status.

## Completion gate

Do not stop until:

- findings are fixed or explicitly bounded,
- relevant checks have evidence,
- Beads is updated,
- commit and push completed.
