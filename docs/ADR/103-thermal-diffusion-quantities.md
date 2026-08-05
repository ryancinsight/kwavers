# ADR 103: Thermal-diffusion physical quantities

- Status: Proposed
- Date: 2026-08-04
- Driver: `KWAVERS-AEQ-MET-66`

## Decision

The public thermal-diffusion parameter contracts use Aequitas quantities:

- perfusion rate: `ReciprocalTime`;
- blood density: `MassDensity`;
- blood specific heat: `SpecificHeatCapacity`;
- arterial temperature: `ThermodynamicTemperature`;
- relaxation time and integration steps: `Time`.

The numerical arrays remain `f64` storage. Conversion to base SI scalars is
performed only where a finite-difference formula, material-model formula, or
Leto storage boundary requires scalar arithmetic. The Plugin trait retains its
host execution contract in seconds and converts once at that boundary; the
thermal solver and physics APIs receive `Time`.

CEM43 is cumulative equivalent minutes at 43 °C, a thermal-dose convention
defined by the Sapareto–Dewey response law. It remains a domain dose value and
is not represented as SI `Time`; the step used to integrate it is typed `Time`.

Eunomia compatibility is unchanged. Thermal quantities are real SI values. If
response code carries real and quadrature components, they remain components
of one existing observable unit; no imaginary SI temperature, time, density,
heat-capacity, or dose unit is introduced.

## Alternatives rejected

- Retaining public `f64` fields would allow seconds, Kelvin, density, and heat
  capacity values to be transposed without a type error.
- Representing CEM43 as `Time` would conflate an equivalent-dose convention
  with elapsed SI time and make thresholds dimensionally ambiguous.
- Adding a compatibility constructor or parallel scalar fields would retain
  the obsolete boundary and duplicate the contract.

## Verification

The implementation must preserve the Pennes source law, Cattaneo–Vernotte
relaxation law, finite-difference update, and CEM43 reference increments.
Tests assert analytical values at one-minute CEM43 and unit-value thermal
updates, and strict package gates verify every migrated direct caller.
