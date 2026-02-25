---
description: "Run a full 3-pass audit-everything wrap-up: investigate, fix, test, document, and close out with evidence."
name: "Audit Everything 3x Wrap-up"
agent: "agent"
---
Perform a rigorous end-of-work wrap-up using **three independent audit passes**, then finish all remediation and verification before handoff.

## Goal

Deliver a production-ready closure pass for the current workspace state by:
1. auditing everything 3 times,
2. fixing in-scope defects,
3. validating with tests,
4. updating existing docs/tests where needed,
5. completing tracker + git closeout.

## Required workflow

### 1) Initialize control plane

- If `.beads/` exists:
  - run `bd prime`
  - find or create an issue for this wrap-up
  - set issue to `in_progress`
- Create and maintain an explicit todo/checklist for this run.

### 2) Investigate with code intelligence first

- Prefer Codanna indexed tools for symbol/context discovery.
- If Codanna MCP is unavailable/empty, use local `codanna` CLI and repository search as fallback.
- For every material claim, capture evidence with exact `file:line` references.

### 3) Run 3 independent audit passes

Use distinct reviewer angles (for example: general consistency, error/root-cause focus, final blocker check).

Each pass must return:
- correctness findings,
- regression risks,
- doc-code mismatches,
- release-readiness status,
- clear blocker vs recommendation split.

Treat “no issues found” as a valid result only if explicitly evidence-backed.

### 4) Remediate findings

- Fix all high-severity in-scope blockers.
- Prefer minimal, targeted edits.
- Reuse existing abstractions; avoid duplicate pathways.
- Update existing tests/docs before adding new files.

### 5) Validate thoroughly

At minimum:
- compile/lint checks for modified files,
- targeted tests for changed behavior,
- broad regression suite.

If any tests are environment/external-network dependent and fail for that reason, classify them explicitly as external blockers with evidence.

### 6) Documentation and audit trail

- Update canonical docs to match actual behavior (math, config names, runtime gating).
- Ensure audit conclusions match current code after remediation.

### 7) Completion gate (must all pass)

Do **not** stop until all of the following are true:
- 3 audit passes completed,
- in-scope blockers fixed or explicitly bounded with evidence,
- relevant tests pass,
- todo/checklist fully resolved,
- beads issue closed,
- `bd sync` run,
- changes committed with Conventional Commit style,
- branch pushed and working tree clean.

## Final response format

Provide a concise report with:
1. what was changed,
2. audit pass summaries (1/2/3),
3. tests run and outcomes,
4. remaining bounded risks,
5. final checklist with all items marked complete.
