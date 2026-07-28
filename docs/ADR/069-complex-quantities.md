# ADR 069: Eunomia Complex Values at Aequitas Physical Boundaries

- Status: Accepted
- Date: 2026-07-28
- Class: [minor]

## Context

Kwavers uses Eunomia `Complex64` for phasors, I/Q samples, spectral arrays,
frequency-domain fields, and impedance calculations. Most of those values are
numerical storage or formula intermediates. Public contracts also carry
physical units in their names and documentation:

- Rayleigh aperture and field pressure phasors are pascals.
- Transducer electrical impedance is ohms.
- Loaded acoustic transmission-line impedance is Rayls.

Aequitas previously supported only real `FloatElement` unit conversion, and it
did not expose electrical impedance as an SI dimension. This forced the public
contracts to return untyped Eunomia complex values.

## Decision

Use Aequitas `Pressure<Complex64>` for Rayleigh aperture and result phasors,
`ElectricalImpedance<Complex64>` for frequency-response and bulk-piezo
electrical impedance, and `AcousticImpedance<Complex64>` for loaded
transmission-line impedance. Reflection coefficients use
`Dimensionless<Complex64>`. Aequitas scales the real and imaginary components
with the same coefficient through Eunomia's provider-owned `UnitScalar`; the
imaginary component is quadrature data, not an independent unit.

Keep dense complex arrays, I/Q samples, dimensionless reflection coefficients,
and complex formula intermediates as explicit numerical boundaries. Do not
wrap them in physical quantities merely because their values are complex.

CFDrs has no public complex physical quantity contract in the audited Womersley
and spectral paths, and Helios has no complex contract; no consumer migration
is required for either repository.

## Verification

- Aequitas tests cover scaled complex conversion, complex dimensional division,
  and zero-sized electrical dimension markers.
- Kwavers transducer tests retain the closed-form complex pressure and
  impedance oracles after typed-boundary migration.
- The remaining complex-value search was classified by public contract versus
  formula/storage boundary in the synchronized Atlas and consumer audits.
