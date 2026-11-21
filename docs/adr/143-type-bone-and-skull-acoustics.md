# ADR 143: Type bone and skull acoustic references

- Status: Proposed
- Class: [major]
- Date: 2026-11-21

The generic bone and skull layers use Aequitas velocity and mass density.
Therapeutic and diagnostic CT mappings share the linear Marsac blend for
sound speed and amplitude attenuation slope. CT maps retain their existing
scalar serialization at the image boundary.

A linear amplitude loss slope has dimension reciprocal length divided by
frequency. Multiplication by a typed frequency gives the reciprocal length
in `p(x) = p(0) exp(−α x)`. To encode dB per centimetre per megahertz,
multiply by `ln(10)/20`, then divide by a centimetre and a megahertz in
base units. Fractional-exponent loss laws require different parameters.

Use `SKULL_MARSAC_ATTENUATION_SLOPE` and `BONE_ATTENUATION_SLOPE` in place of
the scalar constants. The removed hemispherical duplicate shared the Marsac
value. At the existing per-megahertz scalar boundary, evaluate the slope
at the typed 1 MHz reference and extract reciprocal metres.
Analytical tests verify the blend at both endpoints and the midpoint,
linear frequency scaling, and the pressure ratio from the bone dB value.
