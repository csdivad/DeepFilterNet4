---
name: deepfilternet-beads-workflow
description: Beads workflow contract for DeepFilterNet Codex work.
applyTo: "**"
---

# Beads Workflow

If `.beads/` exists, use `bd` for non-trivial work.

## Start

1. Run `bd prime`.
2. Find or create the issue.
3. Set the issue to `in_progress`.

## During

- Record discoveries as notes or linked follow-up issues.
- Keep issue status truthful.
- Do not replace Beads with an ad hoc markdown checklist for multi-stage work.

## End

1. Close completed issues.
2. Run `bd dolt remote list`.
3. If a Dolt remote is configured, run:
   - `bd dolt pull`
   - `bd dolt commit -m "Sync beads state"` when needed
   - `bd dolt push`
4. Push git changes before handoff.

## Reference

- `#file:../../AGENTS.md`
- `#file:../../.github/skills/beads/SKILL.md`
