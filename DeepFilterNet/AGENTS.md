# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` to get the current workflow context.

Prefer `bd prime` for current workflow context. This repo uses repo-managed hook
shims under `.githooks/`, so enable them with `./setup.sh` or
`scripts/install-hooks.sh`.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd dolt remote list   # Check whether a Dolt remote is configured
bd dolt push          # Push Beads state when a Dolt remote is configured
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt status
   bd dolt remote list
   # If a Dolt remote is configured, then:
   bd dolt pull
   bd dolt commit -m "Sync beads state"   # when there are pending Dolt changes
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
- Do **not** run `bd sync`; the current local CLI uses `bd dolt ...` commands instead
