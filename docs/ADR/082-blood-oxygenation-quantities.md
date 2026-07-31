# ADR 082: Aequitas blood-oxygenation quantities

- **Status:** Accepted
- **Date:** 2026-07-31
- **Decision owner:** `KWAVERS-AEQ-MET-43`
- **Scope:** `kwavers-diagnostics::workflows::blood_oxygenation` and its
  runnable photoacoustic example

## Context

The blood-oxygenation workflow exposes optical wavelengths in nanometres,
minimum hemoglobin concentration in molar units, and absorption-reference
coefficients in inverse metres as raw `f64` values. The dense maps are Leto
numerical arrays and are already an explicit storage boundary. Aequitas now
provides the missing semantic molar-concentration and nanometre contracts.

## Decision

Use Aequitas `Length` with `Nanometer` for wavelengths,
`MolarConcentration` with `MicromolePerLiter` for thresholds, and
`ReciprocalLength` for absorption-reference coefficients. Convert to raw
nanometres only when calling the existing optical database and spectral
unmixer APIs, and convert the threshold to the solver's explicit `mol/L`
numerical boundary with Aequitas `MolePerLiter`. Keep dense Leto concentration
and absorption maps as raw numerical arrays because their element type is the
solver/storage contract.

The workflow is real-valued. Eunomia complex values remain confined to any
coherent formula or storage boundary that actually carries them; no imaginary
physical unit is introduced.

## Alternatives

- Retain raw nanometres, molar scalars, and inverse-metre scalars: rejected
  because public contracts would remain unit-ambiguous.
- Reuse Aequitas `NumberDensity`: rejected because amount-of-substance
  concentration and entity number density have distinct semantics despite
  sharing inverse-volume exponents.
- Add consumer-local wrappers: rejected because Aequitas owns the quantity
  vocabulary and unit conversions.
- Type each dense Leto array element: rejected because it would move a
  provider quantity into the numerical storage contract without changing the
  solver's scalar element API.

## Verification plan

- Compile the diagnostics test target and the photoacoustic example.
- Run focused blood-oxygenation Nextest with value-semantic unit, validation,
  and reference-coefficient assertions.
- Run warning-denied Clippy, doctests, RustDoc, formatting, and diff checks.
- Scan the workflow for raw public physical metric fields and verify the
  Eunomia real/complex boundary.

## Verification

The provider Aequitas ADR 0010 and units are accepted and pushed. Kwavers
diagnostics test-target check and the photoacoustic example check pass. The
focused blood-oxygenation Nextest filter passes 3/3 tests with 195 unrelated
tests skipped. Warning-denied all-target Clippy, doctests (1 executable, 5
ignored), RustDoc, formatting, and diff checks pass on 2026-07-31. The only
remaining raw `f64` fields are dense Leto numerical maps; public wavelength,
concentration, and absorption-reference contracts are typed.
