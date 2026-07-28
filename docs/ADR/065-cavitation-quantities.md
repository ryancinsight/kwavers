# ADR 065: Type therapeutic cavitation quantities

- Status: Accepted
- Date: 2026-07-27
- Scope: `kwavers-physics` therapeutic cavitation detector

## Context

The therapeutic cavitation detector accepted frequency, nucleus radius, and
pressure inputs as raw scalars and exposed the Blake threshold and Minnaert
frequency as raw results. Its constructor also accepted a peak-negative-
pressure argument that was never used, so the call contract did not describe
the actual computation.

## Decision

Use Aequitas `Frequency`, `Length`, and `Pressure` at the detector boundary.
Return the Blake threshold and Minnaert resonance as typed values, and accept
typed peak-negative pressure for cavitation index, probability, and regime
classification. Extract base scalars only inside the analytical cavitation
formulas and the existing dense `Array3<f64>` pressure-field detection path.
Keep cavitation index/probability and stable/inertial classifications as
dimensionless or boolean model outputs.

Remove the unused constructor pressure argument rather than retaining a
compatibility parameter. The detector's pressure threshold is computed from
the canonical mechanics cavitation implementation and wrapped as Aequitas
`Pressure` at the provider boundary.

## Alternatives rejected

- Keeping raw scalars preserves dimensional ambiguity at every detector call.
- Retaining the unused constructor parameter would preserve a misleading API
  and hide a real input-contract defect.
- Adding local wrappers would duplicate Aequitas ownership and conversion laws.

## Verification

The `kwavers-physics` test-target check passes. The cavitation-filtered
Nextest run passes 259/259 with 1,298 tests skipped; doctests pass 8/8 with
four ignored; warning-denied Clippy passes; Rustdoc exits successfully with
two pre-existing link warnings; touched-file rustfmt and `git diff --check`
pass.
