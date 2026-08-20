# kwavers-solver

Forward and inverse solvers for [kwavers](https://github.com/ryancinsight/kwavers):
FDTD, PSTD, k-space, Helmholtz, BEM, FWI, RTM, CBS, and PINN.

This crate owns time integration and reconstruction: it advances the fields that
[`kwavers-field`](https://docs.rs/kwavers-field) lays out, over the grid that
[`kwavers-grid`](https://docs.rs/kwavers-grid) defines, using the physics that
[`kwavers-physics`](https://docs.rs/kwavers-physics) specifies. It also declares the
compute-backend trait seam that accelerators implement.

## Forward solvers

| Solver | Method |
|---|---|
| `FdtdSolver` | Finite-difference time domain, staggered grid |
| `PSTDSolver` | Pseudospectral time domain with k-space correction |
| `HybridSolver` | Domain-decomposed FDTD/PSTD |
| `PluginBasedSolver` | Composable physics-plugin pipeline |

Elastic wave equations, Helmholtz, and boundary-element formulations live under `forward`
alongside them.

## Inverse solvers

`Reconstructor` implementations cover time reversal (`TimeReversalReconstructor`),
full-waveform inversion, reverse-time migration, convergent Born series, and
physics-informed neural networks.

## Backend seam

The `backend` module declares the compute-backend and FDTD-accelerator traits. Per the
dependency-inversion rule, this crate depends only on those abstractions — the concrete
WGPU/CUDA implementations live downstream in
[`kwavers-gpu`](https://docs.rs/kwavers-gpu) and are injected at the application boundary.
Adding an accelerator therefore never touches solver code.

## Also here

| Module | Responsibility |
|---|---|
| `workspace` | `ScratchArena` — reusable scratch buffers so time steps stay allocation-free |
| `multiphysics` | `CoupledMultiPhysicsSolver`, field coupling strategies |
| `analytical` | Closed-form references used as solver oracles |
| `interface` | The `Solver` trait and progress-reporting surface |
| `safety` | Solver-level stability and safety guards |
| `factory` / `config` | `SolverConfiguration`, `SolverType`, construction from configuration |

## Documentation

- API reference: <https://docs.rs/kwavers-solver>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
