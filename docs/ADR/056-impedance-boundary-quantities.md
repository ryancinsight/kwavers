# ADR 056: Type impedance-boundary physical inputs

- Status: Accepted
- Date: 2026-07-24
- Scope: `kwavers-boundary::coupling::ImpedanceBoundary`

## Context

`ImpedanceBoundary` exposed target and medium acoustic impedance, the
representative frequency, Gaussian profile frequencies, and custom profile
sample frequencies as `f64` values documented in Rayl or Hz. The same public
boundary already computes the dimensionless impedance ratio and pressure
reflection coefficient from those values. Aequitas provides the matching
`AcousticImpedance` and `Frequency` SI dimensions.

## Decision

Carry acoustic impedance through `ImpedanceBoundary` as
`AcousticImpedance<f64>` and frequency inputs through `Frequency<f64>`. The
`FrequencyProfile` stores typed sample frequencies and accepts a typed
frequency for evaluation. Convert to base SI scalars only inside the existing
Gaussian/interpolation arithmetic and reflection boundary kernel. Keep profile
response, impedance ratio, and reflection coefficient dimensionless because
they are model outputs rather than physical quantities.

The boundary uses `FrequencyProfile::evaluate` as the single interpolation
implementation; `ImpedanceBoundary::impedance_ratio` only applies the typed
impedance ratio to that dimensionless response.

## Alternatives rejected

- Keep scalar fields and rely on unit-bearing names: this preserves the
  dimensional ambiguity that the Aequitas seam exists to remove.
- Use `ThermalConductivity` or another dimensionally similar quantity: the
  semantics are acoustic impedance, so reusing an unrelated quantity would
  make the type contract false.
- Add a scalar compatibility facade: the branch is the migration boundary;
  all in-tree callers are updated and the old public signatures are removed.

## Verification

- Frequency-profile flat, Gaussian, custom interpolation, extrapolation, and
  symmetry tests retain their analytical value oracles with typed inputs.
- Matched, mismatched, rigid, pressure-release, spatial-face, and direction
  reflection tests retain the reflection-coefficient oracles.
- Package checks, Nextest, warning-denied Clippy, doctests, and Rustdoc must be
  rerun after the peer Coeus/Mnemosyne manifest graph allows Cargo to load the
  workspace. The current graph fails before `kwavers-boundary` compilation at
  `D:\atlas\worktrees\coeus\coeus-autograd\Cargo.toml`.
