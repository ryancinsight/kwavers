# ADR 091 — Aequitas electromagnetic SAR quantities

- Status: Accepted
- Date: 2026-07-31
- Owner: `KWAVERS-AEQ-MET-52`

## Context

The electromagnetic material contract exposed electrical conductivity as an
untyped `f64`, and no public Kwavers result represented electromagnetic power
deposition or specific absorption rate. That left the dimensions of
`σ·|E|²` and `σ·|E|²/ρ` implicit at the material and solver seams.

## Decision

Use the Aequitas `ElectricalConductivity` quantity and `SiemensPerMeter` unit
for canonical electromagnetic material data. The distributed conductivity
field is an Aequitas-tagged `DimensionedField` over the existing dense Leto
real storage, so storage remains provider-compatible while the material
contract carries `S/m`.

Add `compute_electromagnetic_deposition` as the canonical consumer operation.
It validates compatible spatial shapes, finite non-negative conductivity,
finite electric-field components, and finite positive mass density, then
returns typed `VolumetricPowerDensity` and `SpecificAbsorptionRate` fields:

```text
q = σ · |E|²       [W/m³]
SAR = q / ρ        [W/kg]
```

The current FDTD electromagnetic fields are real-valued, so no imaginary or
complex physical unit is introduced. If a future Eunomia-backed solver
provides complex phasors, it must reduce them to the Hermitian magnitude at
the numerical formula boundary; the resulting deposition and SAR remain real
power quantities.

## Alternatives rejected

- Raw `f64` conductivity and result arrays: preserve implicit dimensions at a
  public physical contract.
- `ArrayD<ElectricalConductivity>` or `ArrayD<SpecificAbsorptionRate>`: couple
  Leto numerical storage and kernels to domain wrappers instead of retaining
  the existing provider boundary.
- A complex-valued SAR quantity: misrepresents a dissipative real power
  metric; complex support belongs to the phasor input boundary.

## Verification

The provider conductivity dimension is verified by the Aequitas dimension-law
test. Kwavers electromagnetic material tests pass 4/4, electromagnetic solver
tests pass 4/4, and the SAR/EM physics filter passes 26/26. The phase-3d
thermal/optical re-open trigger passes 14/14. Warning-denied Clippy passes for
the affected packages, package doctests pass (3/1/4 executable with ignored
examples), and RustDoc completes with no new crate-local warning. The value
tests cover the Joule/SAR law, zero conductivity, heterogeneous density, and
invalid density. Provider commit `edf746d` supplies the conductivity
vocabulary.
