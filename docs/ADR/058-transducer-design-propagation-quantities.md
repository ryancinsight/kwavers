# ADR 058: Type Transducer Design and Propagation Physical Metrics

- Status: Accepted
- Date: 2026-07-24
- Scope: `kwavers-transducer::design`

## Context

The public aperture-design and focused-propagation contracts use suffixed raw
`f64` fields for dimensions, frequency, sound speed, current, pressure gain,
impedance, intensity, and focal extents. The formulas are input-sensitive and
already covered by value tests, but the API does not prevent dimensional
substitution at construction sites.

## Decision

Use Aequitas quantities for the physical design and propagation fields. Add the
provider-owned pressure-per-electric-current quantity for the propagation gain.
Convert to base SI scalars only inside the existing closed-form arithmetic and
source/grid coordinate adapters. Keep dimensionless flags, ratios, model
coefficients, array counts, and raw coordinate arrays at their existing
contracts.

## Alternatives rejected

- Keep `_m`, `_hz`, `_pa`, and `_w_cm2` scalar names: rejected because names do
  not enforce dimensions.
- Type only the propagation result: rejected because untyped design inputs can
  still admit invalid physical combinations before the calculation.
- Type driver manifest and beam-step serialization in this increment: rejected
  because those are separate serialized/validation contracts and would mix a
  public design cutover with a driver schema migration.
- Add a scalar compatibility facade: rejected because all in-scope callers are
  migrated on the working branch and the old public signatures are superseded.

## Verification

Preserve the existing aperture sizing and focused-pressure analytical value
oracles, including pressure-current scaling, intensity scaling, focal extents,
and dimensionless far-field/grating-lobe flags. Provider dimension laws and
consumer tests must pass. Direct rustfmt and diff checks pass; full workspace
gates remain subject to the peer Coeus graph loading successfully.
