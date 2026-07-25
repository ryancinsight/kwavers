# ADR 059: Type driver beam and result quantities

Status: Accepted

Date: 2026-07-24

## Context

`kwavers-driver` typed the transducer design and focused propagation seams, but
its public beam-step and acoustic/result DTOs still discarded those contracts
into raw SI scalars. This left driver callers and reports able to mix metres,
millimetres, pascals, and watts per square centimetre without a type boundary.

## Decision

Use Aequitas quantities in `KwaversBeamStep`, `KwaversBeamValidation`,
`PressureMap`, and `ExperimentMetrics` for physical geometry, time, pressure,
intensity, focal extents, temperature rise, and thermal headroom. Keep lane
counts, booleans, mechanical index, and model/electrical values at their
existing structural or domain-specific scalar boundaries. Convert once at
manifest text serialization, numerical formula, and check-report boundaries.

## Rejected alternative

Keeping raw DTO fields and documenting units would preserve the existing
conversion hazard and duplicate the typed transducer contract in comments.
Compatibility fields or forwarding accessors are not retained.

## Verification

Preserve the existing beam and experiment value-semantic tests, add typed
boundary assertions where constructors change, run direct rustfmt and diff
checks, then run the focused package gates. The Kwavers workspace gate is
currently subject to the peer Coeus manifest path being restored.
