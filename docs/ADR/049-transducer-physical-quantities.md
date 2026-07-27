# ADR 049: Typed transducer physical quantities

- Status: accepted
- Date: 2026-07-26
- Class: [major]

## Context

`kwavers-transducer::transducers::physics` exposed frequency, geometry,
propagation, wavelength, attenuation, and acoustic-impedance values as raw
`f64` values. Unit suffixes in documentation did not constrain callers and
allowed incompatible SI scales to cross public constructors and result
accessors.

## Decision

- Store frequency and bandwidth as Aequitas `Frequency`.
- Store element, aperture, lens, and propagation distances as `Length`; return
  areas and volumes as `Area` and `Volume`.
- Store acoustic speeds as `Velocity` and acoustic impedances as
  `AcousticImpedance`.
- Store piezoelectric Curie points as `ThermodynamicTemperature` in kelvin;
  catalog values documented in Celsius convert at the constructor boundary.
- Store Rayleigh wavenumber and amplitude attenuation in `ReciprocalLength`;
  integrate them over typed `Length` ranges before the scalar exponential and
  phase kernels.
- Store annular-sector start and span angles as Aequitas `Angle`; extract
  radians only at the Rayleigh quadrature trigonometric boundary.
- Keep normalized magnitude, phase, quality factor, fractional bandwidth,
  directivity, reflection, and insertion-loss results dimensionless. Complex
  surface-pressure phasors remain an explicit numerical-boundary value because
  the current Aequitas public quantity aliases are real-scalar quantities.
- Extract base scalars only at numerical kernels, array/grid rasterization, or
  explicitly formatted reports. No parallel scalar compatibility constructors
  or forwarding accessors remain.

## Alternatives rejected

- Keep raw values with `_hz`, `_m`, or `_pa_s_m` names: names do not enforce
  units at call sites.
- Add typed methods beside the scalar API: this preserves the ambiguity and
  leaves existing callers on the unverified contract.
- Add a local acoustic-impedance or attenuation wrapper: Aequitas already
  owns `AcousticImpedance` and `ReciprocalLength`; duplicating those dimensions
  would fork provider ownership.

## Consequences

This is a pre-release public breaking change. Transducer-design, Rayleigh,
k-Wave, Python, and test callers construct typed physical inputs and receive
typed physical results, including annular-sector angles. The numerical kernels
retain the same equations and scalar layout after explicit boundary extraction.

## Verification

The unchanged KLM bandwidth, geometry identities, Rayleigh propagation,
material matching/reflection, and transducer-design value tests are the
behavioral oracle. The touched package also requires locked check, configured
Nextest, warning-denied Clippy, doctests, Rustdoc, and format verification.
