# ADR 144: Type fractional absorption at a reference frequency

- Status: Proposed
- Class: [major]
- Date: 2026-11-21

A validated power law carries amplitude attenuation in reciprocal metres,
positive reference frequency, and a nonnegative dimensionless exponent.
Only the frequency ratio is raised to a fractional power. Propagation
multiplies the evaluated loss by a length before applying the exponential.
Spectrum preflight evaluates every gain before writing a plane.

The serialized fields remain `alpha_0`, `y`, `f_ref`, and
`dispersion_correction`. Read `alpha_0` in dB/cm at the recorded
frequency. Models using the default 1 MHz reference keep their scale.
The canonical k-Wave conversion continues to emit a scalar spectral
prefactor for CPU and GPU kernels because its units depend on the runtime
exponent.

Tests use a 2 MHz reference and a three-halves exponent. At four times
the reference, attenuation is eight times the stored value. Invalid
frequency in a later plane leaves the entire input spectrum unchanged.
