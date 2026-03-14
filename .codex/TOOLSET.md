# DeepFilterNet Toolset Map

This file documents the tool stack that DeepFilterNet work should prefer. It is
guidance for local agents and skills, not a replacement for Codex runtime tool
registration.

## Core repository investigation

Use these in order:

1. `codanna/get_index_info`
2. `codanna/semantic_search_with_context`
3. `codanna/search_symbols` or `codanna/find_symbol`
4. `codanna/get_calls` and `codanna/find_callers`
5. `codanna/analyze_impact`
6. `rg` only for exact literals, cleanup validation, or index fallback

## Execution and editing

- Shell inspection and commands: `exec_command`
- Multi-command read fanout: `multi_tool_use.parallel`
- File edits: `apply_patch`
- Long-running task state: `update_plan`
- Parallel repo work: `spawn_agent`, `wait`

## Docs and web validation

- Third-party API docs: `context7/*`
- Repository wiki/reference lookups: `deepwiki/*`
- Official web lookups or fresh information: `web` or `brave-search/*`

## Project workflow tools

- Beads CLI through `exec_command`
- Git MCP for status, diff, log, add, commit, push-safe inspection
- Playwright or Puppeteer only when a browser flow is explicitly required

## DeepFilterNet-specific expectations

- `df_mlx` work usually needs both code inspection and doc inspection in
  `docs/` before changes.
- Rust DSP work often spans `libDF`, `pyDF`, `pyDF-data`, and `ladspa`; confirm
  cross-crate impact before edits.
- Release or broad refactor work should run the validator at
  `#file:./scripts/validate_agentic_setup.py` in addition to domain tests.
