# Cavitation and Bubble Dynamics

## Scope

Cavitation chapters cover Rayleigh-Plesset, Keller-Miksis, bubble fields, Blake threshold, Minnaert resonance, Bjerknes coupling, histotripsy, microbubble imaging, and sonoluminescence. Code ownership maps to `kwavers::physics::acoustics::bubble_dynamics`, `kwavers::physics::acoustics::therapy::cavitation`, and `kwavers::domain::sensor::sonoluminescence`.

## Theorem: Minnaert Resonance Scaling

For a small gas bubble of equilibrium radius `R0` in liquid density `rho`, the resonance frequency scales as

```text
f0 proportional_to 1 / R0.
```

### Proof Sketch

Linearizing radial bubble dynamics around `R0` gives an oscillator whose stiffness is set by gas compressibility and whose inertial term scales with liquid density and radius. The angular frequency is proportional to `1/R0`, so the cyclic frequency is also proportional to `1/R0`.

## Algorithm: Cavitation Validation

1. Validate pressure extrema and frequency content before threshold detection.
2. Compute Blake threshold, resonance frequency, and cavitation index from physical parameters.
3. Track bubble radius, velocity, and event statistics as separate outputs.
4. Validate histotripsy and microbubble cases with pressure, spectrum, and event metrics.

## Implementation Targets

- Keep bubble ODE solvers, field coupling, cavitation detection, and optical emission separated.
- Reject threshold tests that assert only that an event exists.
- Preserve event timing and peak pressure in detector outputs.

## Research Anchors

- Histotripsy physical principles: https://doi.org/10.1016/j.ultrasmedbio.2018.10.035
- Histotripsy review: https://doi.org/10.1146/annurev-bioeng-073123-022334
