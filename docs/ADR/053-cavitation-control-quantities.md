# ADR 053: Typed cavitation-control physical quantities

- Status: accepted
- Date: 2026-07-27
- Class: [major]

## Context

The cavitation-control public API represented detector carrier and sample
frequencies, controller response time, safety pressure and temperature limits,
and pulse timing as unit-documented `f64` values. That allowed frequency,
duration, pressure, and temperature values to cross detector, controller,
pulse-sequence, and therapy boundaries without dimensional validation. The
frequency-modulation output also reported the unshifted carrier instead of the
frequency produced by its dimensionless shift.

## Decision

- Store detector carrier/sample rates and controller carrier/output frequency
  as Aequitas `Frequency` values.
- Store controller response time and pulse duration/delay as Aequitas `Time`
  values.
- Store safety pressure and temperature limits as Aequitas `Pressure` and
  `ThermodynamicTemperature` values, using kelvin internally for temperature.
- Keep detector levels, duty cycles, amplitudes, control scores, and frequency
  shifts dimensionless; extract typed values only at numerical formula and
  signal-buffer boundaries.
- Make frequency modulation report the shifted carrier frequency and remove
  the unused sample-rate argument from `PowerModulator::new`.

## Alternatives rejected

- Retain raw fields beside typed fields: rejected because duplicate public
  contracts permit unit drift and require synchronization.
- Add Kwavers-owned frequency, timing, pressure, or temperature wrappers:
  rejected because Aequitas already owns these SI dimensions.
- Keep the unused `PowerModulator` sample-rate argument: rejected because it
  falsely signals a configuration dependency that the implementation ignores.

## Consequences

This is a pre-release public breaking change to cavitation-control detector,
controller, pulse-sequence, safety, and therapy-call signatures. Scalar
extraction remains explicit at numerical kernels, while physical values stay
typed through the public contracts. The frequency-modulation result now has
the same physical meaning as the generated carrier signal.

## Verification

Value-semantic tests cover typed detector defaults, controller output
frequency, safety-temperature conversion, and pulse-sequence timing. Package
checks, focused Nextest, doctests, Rustdoc, Clippy, and touched-file format
results are recorded in the child gap audit for the exact delivery commit.
