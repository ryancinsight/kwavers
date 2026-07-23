# ADR 047: Typed pulsed-laser physical quantities

- Status: accepted
- Date: 2026-07-23
- Class: [major]

## Context

`kwavers-physics::electromagnetic::PulsedLaser` represented optical power,
pulse timing, repetition rate, wavelength, beam radii, peak fluence, and
average power as raw `f64` values. Unit suffixes existed only in comments, so a
caller could pass incompatible scalar units and the public result erased the
physical dimension.

## Decision

- Store peak optical power as Aequitas `Power`.
- Store pulse duration and beam radii as Aequitas `Time` and `Length`.
- Store repetition rate as Aequitas `Frequency` and wavelength as `Length`.
- Return peak fluence as `EnergyPerArea` and average power as `Power`.
- Express pulse energy, beam area, and fluence equations through Aequitas
  dimensional arithmetic; retain the Gaussian, flat-top, and Bessel model
  factors and their existing closed forms.

The constructor and beam-profile fields are migrated in the same change. No
scalar compatibility constructor or forwarding methods remain; the only
in-repository caller is updated in the module's value-semantic tests.

## Alternatives rejected

- Retain raw scalars with `_w`, `_s`, or `_m` naming: rejected because names do
  not enforce units at the call site.
- Add a typed result beside the scalar result: rejected because it preserves two
  contracts and leaves the public scalar boundary ambiguous.
- Add a new local wavelength or fluence type: rejected because Aequitas already
  owns the required SI dimensions.

## Consequences

This is a pre-release public breaking change. Callers construct typed optical
parameters and receive typed energy metrics. The photoacoustic equations remain
the same, while the type system prevents unit mixing at the source boundary.
