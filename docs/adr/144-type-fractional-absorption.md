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

## Accepted dispersion extension, 2026-11-22

Medium 7.0.0 uses the reference-frequency power law for both dispersion
entry points. A second scalar angular-frequency prefactor cannot encode
its units independently of the runtime exponent. Clients specify the loss
at a reference frequency instead.

For positive angular frequency ω, define q = c₀ α(ω) tan(πy/2)/ω.
Then k = ω(1+q)/c₀ and dk/dω = (1+yq)/c₀. Their inverses give phase
and group velocity. A zero group slowness is singular. Negative group
slowness gives signed group velocity on the positive-phase branch. At DC,
use the right-hand limit. Even exponents have zero tangent, odd exponents
are poles in this particular power law. Zero loss and disabled dispersion
give c₀ at all frequencies.

The k-space operator takes signed angular wavenumbers in radians per metre
and a time step. It evaluates the phase at the same reference frequencies
as the original operator, validates the whole update, and then rotates
the complex spectrum. Verify the derivative, DC limits, signs, opposite
wavenumbers and atomic rejection against analytic values.

## Accepted physical fractional derivative, 2026-11-22

Both methods take a homogeneous, dimensioned signal and return a new
Aequitas runtime quantity. If the input has SI exponents D, the result
has exponents D − α·time. Use the exact binary value of the supplied
dimensionless order to derive that metadata. Each derivative can consume
a previous result. Time samples use a finite positive timestep in seconds.
The causal Grünwald–Letnikov convolution has zero prehistory and includes
the present sample plus at most the specified number of lags.

Spectral bins use signed frequencies in Hz and the principal multiplier
(i·2πf)^α. Every coefficient needs a corresponding finite frequency.
Repeated or unsorted frequencies are valid. A DC bin has multiplier zero
for positive α and one for zero α. Negative α at DC is singular and
returns an error. Borrow the input so every rejected call preserves it.
The periodic Fourier derivative and the causal finite-history operator
have different boundary conditions.
