# `pykwavers` Inventory

## Overall status

`pykwavers` is intended to be a thin PyO3 binding layer over canonical Rust APIs.
Current state is mixed.

## Canonical candidate responsibilities

- expose canonical Rust solver/config types
- orchestrate parity runs from Python
- provide importable CPU/reference simulation surface

## Current structural issues

- `src/lib.rs` is monolithic
- default feature set previously included `gpu`, which made default builds depend on the broken
  GPU tranche
- public API likely mixes canonical and legacy ownership

## Confirmed tranche-one state

- `cargo check -p pykwavers --no-default-features` passes
- minimal CPU binding path is viable
- GPU-enabled path is not viable yet and is deferred

## Immediate tranche-one priorities

1. keep default binding path CPU/minimal
2. split `src/lib.rs` into binding modules
3. expose canonical photoacoustic and acoustic-core APIs only
4. add Rust/Python parity checks per exposed surface
