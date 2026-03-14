---
name: deepfilternet-release-closeout
description: Release and session-closeout checklist for DeepFilterNet.
applyTo: "**"
---

# Release and Closeout Overlay

No session is complete until the repository state is pushed.

## Closeout checklist

1. Run relevant quality gates.
2. Update Beads status and file follow-up issues for unfinished work.
3. Review git status for unrelated changes and explain your decision.
4. Commit with a Conventional Commit message.
5. Run:
   - `git pull --rebase`
   - `bd dolt status`
   - `bd dolt remote list`
   - `bd dolt pull`, `bd dolt commit`, `bd dolt push` when configured
   - `git push`
6. Confirm the working tree and branch status.

## Repo-local validation

Run the agentic validator when this `.codex` setup changes:

```bash
python3 .codex/scripts/validate_agentic_setup.py
```
