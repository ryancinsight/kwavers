# ADR 066: Type lithotripsy configuration quantities

- Status: Accepted
- Date: 2026-07-27
- Scope: `kwavers-physics` lithotripsy configuration

## Context

`LithotripsyConfig` exposed shock-wave peak pressure, pulse duration, and
repetition rate as unit-documented `f64` fields. The module does not yet own
the shock-wave, stone-fracture, cavitation-cloud, or bioeffects solvers, so the
metric closure must remain limited to the existing configuration boundary.

## Decision

Use Aequitas `Pressure`, `Time`, and `Frequency` for the three configuration
fields and their defaults. Future solver implementations consume the typed
configuration and extract base values only at their numerical kernels.

The solver components listed in the module documentation remain separate
implementation work; this ADR does not claim those components exist.

## Alternative rejected

Retaining raw scalars would preserve dimensional ambiguity in the only current
public lithotripsy contract. Adding local wrappers would duplicate Aequitas
ownership without a second implementation requirement.

## Verification

The `kwavers-physics` test-target check, focused therapy tests, warning-denied
Clippy, doctests, Rustdoc, rustfmt, and `git diff --check` pass.
