# ADR 074: Type Bulk Piezoelectric Quantities

- Status: Accepted
- Date: 2026-07-31
- Driver: `KWAVERS-AEQ-MET-36`

## Decision

The bulk piezoelectric resonator stores thickness as `Length`, electrode area
as `Area`, density as `MassDensity`, stiffened modulus as `Pressure`, and
relative permittivity and thickness coupling as `Dimensionless`. Derived
sound speed, antiresonance and series-resonance frequencies, capacitances,
resonance gap, electrical frequency inputs, and matching-layer geometry use
the corresponding Aequitas `Velocity`, `Frequency`, `Capacitance`, and
`Length` contracts.

Formula evaluation extracts canonical SI scalars only at the IEEE thickness-
mode relation, the resonance bisection, and the electrical/acoustic
transmission-line boundaries. The public API does not encode units in names
or retain raw physical scalars as an alternate path.

Eunomia `Complex64` remains at the phasor boundaries: electrical impedance,
loaded acoustic impedance, and reflection coefficient. The real and
quadrature components have the same Aequitas physical dimension. An
imaginary or complex unit is not a physical SI dimension and is therefore not
added to Aequitas.

## Rejected alternative

Retaining raw scalar geometry and adding conversions at each resonator or
matching-layer caller would preserve dimensional ambiguity and allow metres,
hertz, and metres-per-second values to be interchanged without a type error.
Introducing a separate imaginary unit would also misrepresent complex
phasors, whose components share the underlying physical unit.

## Verification

The analytical tests cover inverse-thickness scaling, sound-speed and
capacitance formulas, resonance/coupling round trips, coupling-gap
monotonicity, reactive electrical impedance, series and antiresonance limits,
and half-/quarter-wave transmission-line identities. `kwavers-transducer`
package check, Nextest (219/219 with one skip), warning-denied Clippy,
doctests (2/2 with six ignored), and formatting pass.
