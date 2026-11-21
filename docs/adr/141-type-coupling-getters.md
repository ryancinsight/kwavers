# ADR 141 · Carry physical quantities in coupling getters

- Status: Proposed
- Class: [major]
- Date: 2026-11-21

Coupling points carry three length components. Mechanical stress, velocity,
and acoustic impedance carry their respective dimensions. Energy exchange,
viscous dissipation, and perfusion carry volumetric power. Thermal laws take
absolute temperatures and use their difference for perfusion. Hyperion owns
the distinction between absorption and reduced scattering. Callers convert
at array or external scalar boundaries.

Compute the acoustic pressure coefficients from impedances normalized by
their larger value. For positive finite impedances the denominator is
between one and two. The common factor cancels and a matched interface
gives zero reflection and unit transmission at any representable scale.
Use material getters for thermal properties and require a model-provided
viscous dissipation law.

The source cohort is 4.0.0. Implementations migrate getter return types and
take length-typed positions. Acoustic coefficient callers supply samples on
both sides. The coupling tests assert orientation and extreme matched
impedances, and the compiler rejects mixing time with volumetric power.
