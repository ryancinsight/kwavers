# ADR 049: Typed transducer frequency and geometry quantities

- Status: accepted
- Date: 2026-07-23
- Class: [major]

## Context

`kwavers-transducer::transducers::physics` exposed element dimensions,
frequency-response metrics, sampled frequency points, sound speed, area,
volume, and pulse-resolution results as raw `f64` values. Unit comments did
not prevent callers from mixing metres, millimetres, hertz, or metres per
second at the public boundary. The same raw values propagated through
`TransducerDesign`.

## Decision

- Store `ElementGeometry` dimensions, pitch, and kerf as Aequitas `Length`.
- Return element area and volume as Aequitas `Area` and `Volume`.
- Accept Aequitas `Velocity` and return Aequitas `Frequency` for thickness and
  lateral resonance calculations.
- Store frequency-response center frequency, bandwidths, and sampled
  frequencies as Aequitas `Frequency`.
- Accept typed frequency and sound-speed inputs for KLM construction, pulse
  characteristics, and sensitivity sampling; return pulse length and axial
  resolution as Aequitas `Length`.
- Retain dimensionless fractional bandwidth, quality factor, magnitude, phase,
  directivity factors, and reflection/insertion-loss values as scalars. Keep
  complex electrical impedance scalar because Aequitas currently provides
  acoustic impedance, not an electrical-ohm dimension.

The design module converts only at the existing downstream scalar boundaries
for directivity, matching-layer, sensitivity, capacitance, and compliance
models whose contracts remain scalar. No scalar compatibility constructors or
parallel result fields remain.

## Alternatives rejected

- Preserve raw fields with unit-suffixed names: rejected because naming does
  not enforce dimensional correctness.
- Add typed accessors beside raw public fields: rejected because it retains two
  contracts and leaves unit mixing possible.
- Treat complex electrical impedance as acoustic impedance: rejected because
  the dimensions and model role differ; the provider gap is recorded in the
  transducer audit instead.

## Consequences

This is a pre-release public breaking change to the transducer physics design
API. In-repository callers construct typed frequency and length quantities and
receive typed physical metrics. KLM, geometry, focal-resolution, and mode
separation equations retain their value semantics, covered by the transducer
unit and integration tests. Rayleigh propagation and materials remain the
next bounded transducer audit slices.
