# Photoacoustics

## Scope

Photoacoustics covers optical absorption, initial pressure generation, acoustic propagation, line reconstruction, time reversal, dual-modal PA/US workflows, and skull-induced aberration. Code ownership maps to `kwavers::simulation::modalities::photoacoustic`, `kwavers::solver::photoacoustics`, `kwavers::physics::optics`, and reconstruction modules.

## Theorem: Initial Pressure from Optical Absorption

For absorbed optical energy density `H` and Gruneisen parameter `Gamma`, the initial acoustic pressure is

```text
p0 = Gamma H.
```

### Proof Sketch

Under stress and thermal confinement, deposited heat produces an isochoric temperature rise. The thermoelastic pressure rise is proportional to absorbed energy density through the dimensionless Gruneisen parameter.

## Algorithm: Photoacoustic Validation

1. Validate optical fluence and absorption maps.
2. Compute `p0` with explicit `Gamma`.
3. Propagate acoustically with the same medium and sensor contract used by ultrasound.
4. Validate reconstruction against analytical geometry or cached k-Wave reference traces.

## Implementation Targets

- Keep optical and acoustic solvers coupled through `p0`, not through hidden shared state.
- Preserve dual-modal PA/US raw data separately before fusion.
- Validate line reconstruction ordering under transpose and interpolation edge cases.

## Research Anchors

- Deep-tissue PACT review: https://www.nature.com/articles/s44303-024-00048-w
- Dual-modal PA/US review: https://doi.org/10.3389/fphot.2024.1359784
