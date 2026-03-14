---
name: DeepFilterNet Training Run Readiness
description: Validate that a df_mlx training-related change is coherent, documented, and ready for a real run.
agent: dfn-shadow-orchestrator
---

Use this prompt for `df_mlx` training or run-config work.

## Workflow

1. Confirm the active entrypoint and config layers.
2. Read matching docs in `docs/` and `DeepFilterNet/df_mlx/README.md`.
3. Review checkpoint, loss, VAD, and preset implications as applicable.
4. Run the smallest meaningful test or command checks.
5. Update docs if any config or runtime semantics changed.

## Required output

- exact files and docs touched,
- command evidence,
- remaining runtime caveats.
