---
name: deepfilternet-release-readiness
description: DeepFilterNet closeout workflow covering Beads, tests, docs, commits, and push.
---

# DeepFilterNet Release Readiness

Use this skill when preparing to hand off substantial work or ship a repo-wide
change.

## Workflow

1. Load `#file:../../instructions/release.instructions.md`.
2. Run the domain quality gates relevant to the touched files.
3. Run the agentic validator if `.codex` assets changed.
4. Review unrelated local modifications and state whether they are preserved or
   included.
5. Update or close Beads issues.
6. Commit, push, and confirm status.

## Required evidence

- tests or command results,
- git status before and after closeout,
- Beads sync result when configured,
- bounded remaining risks.
