---
name: deepfilternet-rust-dsp-instruction
description: Rust DSP and bindings overlay for DeepFilterNet.
applyTo: "{libDF,pyDF,pyDF-data,pyDF-augment,ladspa}/**/*.{rs,toml}"
---

# Rust DSP Overlay

Use this for Rust crates and binding layers.

## Scope

- `libDF/`: DSP/runtime core
- `pyDF/`, `pyDF-data/`, `pyDF-augment/`: bindings and augmentation/data crates
- `ladspa/`: plugin packaging and runtime integration

## Working rules

1. Confirm cross-crate impact before edits.
2. Preserve workspace conventions from the root `Cargo.toml`.
3. Prefer `cargo fmt`, `cargo build`, and `cargo test` for validation.
4. If bindings change, confirm whether the matching Python package or docs need
   updates.

## Common commands

- `cargo build`
- `cargo test`
- `cargo fmt`
- `maturin develop --release -m pyDF/Cargo.toml`
- `maturin develop --release -m pyDF-data/Cargo.toml`
