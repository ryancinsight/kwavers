# ADR 142: Apply local material coefficients in field coupling

- Status: Proposed
- Class: [minor]
- Date: 2026-11-21

The collocated pressure, light, and thermal fields share a cell index. A
medium provider resolves that cell at the caller's acoustic frequency. The
provider supplies the signed photoelastic derivative, optical absorption,
acoustic intensity absorption, density, sound speed, and specific heat.
The response rate converts refractive-index change into relative intensity
change over the elapsed time. Acoustic amplitude attenuation must be
converted to intensity attenuation before it enters the material because
intensity is proportional to squared amplitude.

The weak pass computes `I' = I(1 + s β p k Δt)` and
`ΔT = (μₐ I' + α p²/(2ρc)) Δt/(ρ cₚ)`. The gains keep their
dimensions through the field update. These inputs can describe a constant
material or a spatial and frequency-dependent law. The local solver uses
the same gains for subsequent passes of a strong or adaptive step.
Preflight checks the shape, time, frequency, and every coefficient before
mutating a field. A derived denominator outside the representable
positive range fails preflight.

Embedding optical or acoustic parameters as nominal values in the
medium-specific API would discard the caller's material law. Adding
optical properties to all general media would require unrelated
implementations to invent coefficients. The existing nominal entry
points retain the historical nominal values. The fixed acoustic intensity
loss is 0.5 m⁻¹ and is defined independently of the tissue spectral slope
in dB/(cm·MHz).

Tests use a real coupled solver and compare two adjacent cells across two
frequencies against the analytical result. An overflowing denominator in
the second cell verifies that invalid material cannot partially apply.
