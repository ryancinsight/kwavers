# ADR 072: Aequitas contracts for GPU/CPU equivalence metrics

Status: Accepted and implemented under `KWAVERS-AEQ-MET-34`.

## Context

`kwavers-gpu` kept peak pressure and measured CPU/GPU execution times as raw
`f64` fields in `EquivalenceReport`. The dense `LetoArray3<f64>` samples are a
deliberate numerical-storage boundary, but a public report is a physical metric
boundary and must distinguish pressure from elapsed time. Relative error,
tolerance, divergent fraction, and speedup are dimensionless.

## Decision

Use Aequitas `Pressure<f64>`, `Time<f64>`, and `Dimensionless<f64>` in the
equivalence validator and report. Store time canonically in seconds and format
it in milliseconds only at the display boundary. Keep scalar extraction inside
the dense-array reduction and comparison formulas. Use Eunomia's real
`UnitScalar` implementation through Aequitas; this validation path has no
complex or imaginary physical quantity.

The former unit-suffixed report fields are replaced in place and all repository
callers and tests migrate in the same change. No compatibility fields or
wrappers remain.

## Alternatives rejected

- Keep raw report fields: rejected because callers can interchange pressure and
  duration values without a type error.
- Type every dense array element: rejected because it changes the storage and
  GPU-provider boundary without adding contract value; `DimensionedField` and
  the existing Leto boundary retain the zero-copy numerical representation.
- Introduce a complex pressure: rejected because this validator compares real
  time-domain fields and has no phasor or quadrature contract.

## Verification

- `cargo check -p kwavers-gpu --offline`
- `cargo nextest run -p kwavers-gpu --offline`
- `cargo clippy -p kwavers-gpu --all-targets --offline -- -D warnings`
- `cargo test --doc -p kwavers-gpu --offline`
- targeted rustfmt and raw unit-suffixed public-field scans
