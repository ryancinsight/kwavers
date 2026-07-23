# ADR 050: Typed transducer materials and Rayleigh propagation quantities

- Status: accepted
- Date: 2026-07-23
- Class: [major]

## Context

The transducer material and lens APIs still exposed acoustic impedance, density,
sound speed, Curie temperature, lens dimensions, wavelength, delay, and
correction thickness as `f64` values. Rayleigh propagation likewise accepted
wavenumber, attenuation, layer thickness, and range as untyped values. Unit
comments did not prevent mixing metres with millimetres, Rayl with MRayl, or
reciprocal metres with metres at the public boundary.

## Decision

- Store piezoelectric density, sound speed, acoustic impedance, and Curie
  temperature as Aequitas `MassDensity`, `Velocity`, `AcousticImpedance`, and
  `ThermodynamicTemperature`.
- Store backing/matching impedances and thicknesses as `AcousticImpedance` and
  `Length`; accept typed frequency and impedance in matching-layer design.
- Store acoustic-lens and Fresnel-zone dimensions as `Length`, accept typed
  sound speed, and return typed focal lengths, delays, zone radii, and
  corrective-lens thicknesses. Retain f-number, coupling, Q, attenuation
  coefficients, and reflection/transmission coefficients as model scalars.
- Store Rayleigh wavenumber and attenuation as `ReciprocalLength`, accept layer
  thickness and propagation range as `Length`, and retain accumulated phase,
  attenuation exponent, and coherent complex pressure as scalar mathematical
  values.
- Convert only at the PyO3 boundary and at legacy rasterizer/model boundaries;
  no parallel scalar compatibility constructors or result fields remain.

## Alternatives rejected

- Preserve unit-suffixed `f64` fields: rejected because naming does not enforce
  dimensional correctness.
- Add typed accessors beside raw public fields: rejected because it retains two
  contracts and leaves unit mixing possible.
- Treat attenuation, f-number, coupling, or pressure phasors as ordinary SI
  quantities: rejected because they are model coefficients, dimensionless
  mathematical accumulations, or complex-valued boundary values rather than
  currently supported Aequitas scalar dimensions.

## Consequences

This is a pre-release public breaking change to the transducer materials and
Rayleigh propagation APIs. In-repository Rust and PyO3 callers now construct
typed physical values and convert only when crossing a scalar storage or Python
array boundary. Material lensmaker, Fresnel, isoplanatic, corrective-phase, and
Rayleigh propagation value oracles remain unchanged and pass the full
`kwavers-transducer` library suite (221/221, one skipped). Rayleigh aperture
geometry coordinates and observation-point arrays remain the next bounded
transducer boundary; they are tracked separately from this propagation slice.
