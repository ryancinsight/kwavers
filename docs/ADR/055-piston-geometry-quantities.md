# ADR 055: Type basic piston geometry with Aequitas

## Context

`kwavers-transducer::basic::PistonConfig` accepted centre coordinates,
diameter, and Gaussian apodization sigma as raw `f64` values. The neighbouring
Rayleigh aperture API already owned typed lengths and validated
`CartesianPosition`, so the basic piston surface was an inconsistent physical
boundary.

## Decision

Store piston centre as the shared `CartesianPosition`, diameter and radius as
Aequitas `Length`, and Gaussian sigma as `Length`. `PistonBuilder` and the
source accessors use the same typed contract. Convert to raw metres only inside
the apodization and `kwavers_source::Source` coordinate boundary, which is the
existing scalar grid contract. The factory validates the domain position before
constructing the typed piston configuration.

## Alternatives rejected

- Keep raw piston geometry: leaves a sibling physical API untyped.
- Add a piston-only position wrapper: duplicates the existing validated
  `CartesianPosition` owner.
- Type the entire `Source` trait: expands this change into a provider boundary
  unrelated to the piston metric gap.

## Verification

The transducer unit regression constructs a typed centre, diameter, and Gaussian
sigma, verifies the SI values returned by the source, and checks unit-weight
Gaussian apodization at the centre. Focused package gates remain pending until
the shared Coeus/Mnemosyne provider graph resolves before Kwavers compilation.
