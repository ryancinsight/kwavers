# ADR 099 — Aequitas transducer design and propagation quantities

Status: Accepted — 2026-08-03

## Context

The aperture-design and focused-propagation contracts still exposed physical
dimensions, operating conditions, drive quantities, and derived beam metrics as
raw scalars. The direct consumers are the Kwavers driver validation and
experiment adapters. Their legacy report structures intentionally serialize
pressure, intensity, and beam widths as Pa, W/cm², and mm.

## Decision

Use Aequitas quantities at the transducer contract:

- `Length` for aperture, pitch, kerf, element extent, wavelength, coordinates,
  and beam widths.
- `Frequency` for operating frequency and `Velocity` for sound speed.
- `Dimensionless` for pitch/kerf fractions, fill factor, and mechanical index.
- `ElectricCurrent`, `PressurePerElectricCurrent`, and `AcousticImpedance` for
  the focused drive contract.
- `Pressure` and SI `Intensity` for focused propagation outputs.

Extract base scalars only at Euclidean, trigonometric, coherent-propagation,
mesh, explicit unit-conversion, and legacy driver serialization boundaries.
Driver report DTOs retain their established Pa/W/cm²/mm fields and perform the
conversion at that boundary; they do not feed raw values back into the
transducer contract.

## Eunomia compatibility

The coherent pressure calculation uses local real and quadrature accumulators
for one observable pressure signal. If those values cross an Eunomia FFT or
storage boundary, both components retain the same observable unit. The
quadrature component is not an imaginary SI dimension, so no imaginary-unit
quantity or complex-valued physical wrapper is introduced.

## Verification

Analytical geometry and coherent-propagation regressions cover realized pitch,
element/channel positions, focal pressure, intensity, mechanical index, and
beam widths. Invalid typed inputs cover non-finite focus and zero drive
current. The affected transducer and driver suites pass through Nextest;
warning-denied Clippy, formatting, diff, public-raw-signature, and complex
boundary scans pass. The repository-owned hosted matrix remains the merge
gate for the final branch head.

Refs: `KWAVERS-AEQ-MET-60` in `backlog.md`.
