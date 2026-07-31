# ADR 081: Aequitas plane-wave compounding quantities

- **Status:** Accepted
- **Date:** 2026-07-31
- **Decision owner:** `KWAVERS-AEQ-MET-42`
- **Scope:** `kwavers-diagnostics::workflows::plane_wave_compounding`

## Context

The plane-wave compounding workflow exposes angles, frequency, sound speed,
aperture geometry, sampling lengths, dynamic range, and frame-rate estimates
as raw scalars even though Aequitas provides the corresponding dimensions.
Its coherent images are Eunomia complex numerical arrays, not quantities with
an imaginary physical unit.

## Decision

Use Aequitas `Angle`, `Frequency`, `Velocity`, `Length`, and `Dimensionless`
for the public configuration and frame-rate result. Store the internal
wavelength, wave number, angular frequency, and generated angles with their
corresponding Aequitas dimensions. Extract SI scalars only at numerical
formula, mesh/solver, and display/report boundaries. Keep Eunomia complex
values at the coherent-image storage and phase-formula boundaries.

## Alternatives

- Retain raw SI scalars: rejected because callers can pass incompatible
  physical dimensions and the public contract remains unit-ambiguous.
- Add local wrapper types: rejected because Aequitas is the provider-owned
  quantity SSOT and a consumer wrapper would duplicate its dimension model.
- Introduce a complex physical unit: rejected because plane-wave angles,
  timing, geometry, and display metrics are real-valued; complex values are
  already handled by Eunomia at the coherent numerical boundary.

## Verification plan

- Compile the diagnostics package test target.
- Run the focused plane-wave Nextest filter and assert value-semantic angle,
  geometry, frame-rate, field, beamforming, and thermal-boundary behavior.
- Run warning-denied Clippy, doctests, RustDoc, formatting, and diff checks.
- Scan the workflow for remaining raw public physical metric fields and verify
  that complex storage remains at the Eunomia boundary.

## Verification

The package test-target check passes with `cargo check -p
kwavers-diagnostics --tests --offline`. The focused Nextest filter passes
10/10 tests with 185 unrelated tests skipped. Warning-denied all-target
Clippy, doctests (1 executable, 5 ignored), RustDoc, formatting, and diff
checks pass. The source scan finds no raw public physical metric fields in the
plane-wave contract; scalar extraction is confined to formula, mesh/solver,
and display/report boundaries. Shared unused-provider-patch and linker
warnings remain outside this metric family.
