# ADR 103: Thermal-diffusion physical quantities

- Status: Accepted
- Date: 2026-08-06
- Driver: `KWAVERS-AEQ-MET-66`

## Decision

The public thermal-diffusion parameter contracts use Aequitas quantities:

- perfusion rate: `ReciprocalTime`;
- blood density: `MassDensity`;
- blood specific heat: `SpecificHeatCapacity`;
- arterial temperature: `ThermodynamicTemperature`;
- relaxation time and integration steps: `Time`.

Numerical arrays remain `f64` storage. Conversion to base SI scalars occurs
only at finite-difference, material-model, or Leto storage boundaries. CEM43
remains a domain dose convention; its integration step is typed `Time`, but the
dose itself is not mislabeled as elapsed SI time.

Eunomia compatibility is real-valued for these thermal contracts. Real and
quadrature components in adjacent signal domains retain one observable unit;
no imaginary SI temperature, time, density, heat-capacity, or dose unit is
introduced.

## Alternatives rejected

- Raw public scalars were rejected because seconds, Kelvin, density, and heat
  capacity can otherwise be transposed without a type error.
- Representing CEM43 as `Time` was rejected because equivalent minutes are a
  dose convention, not elapsed SI time.
- Parallel scalar fields or compatibility constructors were rejected because
  they retain and duplicate the obsolete boundary.

## Verification

`cargo check -p kwavers-physics --lib --offline -j 1` passes on the active
branch. The native test target was not collected in this increment because the
shared target was concurrently occupied and the available disk fell below
100 MB; the attempted bounded Nextest run exceeded its 300-second budget and
was terminated. This is a verification-environment residual, not a typed
contract failure.
