---
name: deepfilternet-rust-dsp
description: Rust DSP, bindings, and plugin workflow guidance for DeepFilterNet workspace crates.
---

# DeepFilterNet Rust and DSP Workflow

Use this skill when a task touches Rust crates or Python bindings backed by
Rust.

## Required references

- `#file:../../instructions/rust-dsp.instructions.md`
- `Cargo.toml`

## Workflow

1. Identify the owning crate and downstream dependents.
2. Trace call impact across `libDF`, bindings, and plugin code.
3. Keep changes narrow and workspace-consistent.
4. Run `cargo fmt` and the smallest relevant build/test commands.
5. If bindings or packaging change, check adjacent Python or docs impact.
