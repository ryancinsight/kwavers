# Acoustic Propagation

![Wave energy flow](figures/wave_energy_flow.svg)

## Scope

Propagation chapters cover first-order acoustics, second-order wave equations, heterogeneous media, absorption, dispersion, boundary layers, and source injection. Code ownership maps to `kwavers::solver::forward::fdtd`, `kwavers::solver::forward::pstd`, `kwavers::domain::boundary`, and `kwavers::domain::medium`.

## Theorem: Linear Acoustic Wave Equation

For a homogeneous inviscid medium, the first-order equations

```text
partial_t p = -rho c^2 div(u)
partial_t u = -(1/rho) grad(p)
```

imply

```text
partial_tt p - c^2 Delta p = 0.
```

### Proof Sketch

Differentiate the pressure equation in time and substitute `partial_t u` from the momentum equation. Constant `rho` and `c` commute with spatial derivatives, giving `partial_tt p = c^2 Delta p`.

## Algorithm: Propagation Validation

1. Validate grid spacing, CFL, medium parameters, and boundary thickness.
2. Run a plane-wave or point-source analytical case.
3. Compare phase velocity, amplitude decay, and boundary reflection against the analytical expectation.
4. Preserve full source and sensor ordering when comparing with k-wave-python.

## Implementation Targets

- Keep FDTD and PSTD propagation contracts separate from source geometry and sensor sampling contracts.
- Route FFT work through Apollo only.
- Preserve lower-dimensional embeddings as 3-D tensors with inactive-axis derivatives equal to zero.

## Research Anchors

- Treeby and Cox k-Wave model: http://www.k-wave.org/
- k-wave-python documentation: https://k-wave-python.readthedocs.io/
- k-Wave 2.0 broadband pseudospectral development context: https://ccmi-cdt.org/phd_projects/entries/Cox_kwave.html
