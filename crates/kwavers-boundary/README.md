# kwavers-boundary

Boundary conditions for [kwavers](https://github.com/ryancinsight/kwavers): CPML/PML
absorbing layers, FEM/BEM boundary managers, multiphysics coupling, periodic boundaries,
and smoothing.

A finite computational domain has to stand in for an infinite medium. This crate owns how
that truncation is handled — absorbing layers for time-domain solvers, constraint assembly
for variational solvers, and the interface conditions that couple dissimilar subdomains.

## Boundary families

### Time-domain (FDTD, PSTD)

Applied during time stepping to suppress reflections from the domain edge:

- `DomainPMLBoundary` — perfectly matched layer with per-physics (acoustic/optical)
  damping profiles
- `CPMLBoundary` — convolutional PML, including per-dimension profiles and dispersive
  parameters, for grazing-incidence and evanescent-wave accuracy
- `PmlExpFactors` — precomputed `exp(-σ Δt/2)` factors so the split-field update costs
  O(N) multiplications per step instead of O(N) transcendental evaluations

### Variational (FEM, BEM)

Applied during matrix assembly rather than during time stepping:

- `FemBoundaryManager` — Dirichlet, Neumann, Robin, and radiation conditions
- `BemBoundaryManager` — boundary-integral conditions for the H/G matrix pair

### Interface and periodic

- `PeriodicBoundaryCondition` — wrap-around domains
- `ImpedanceBoundary`, `MaterialInterface`, `SchwarzBoundary`, `AdaptiveBoundary` —
  multiphysics and domain-decomposition coupling
- `smoothing` — edge smoothing to avoid spurious high-wavenumber content

## Example

PML configuration is validated at construction; an unphysical layer is rejected with a
typed error rather than silently producing reflections.

```rust
use kwavers_boundary::{DomainPMLBoundary, DomainPmlConfig};

// 20-cell absorbing layer, defaults for the remaining damping parameters.
let config = DomainPmlConfig { thickness: 20, ..Default::default() };
let boundary = DomainPMLBoundary::new(config).unwrap();

// A zero-thickness layer absorbs nothing and is refused.
let invalid = DomainPmlConfig { thickness: 0, ..Default::default() };
assert!(DomainPMLBoundary::new(invalid).is_err());
```

## Architecture note

Runtime boundary selection lives in `kwavers-simulation`; this crate holds the boundary
implementations themselves and depends only on `kwavers-core`, `kwavers-grid`, and the
array substrate.

## Documentation

- API reference: <https://docs.rs/kwavers-boundary>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
