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

For single-bubble gas exchange, opt in to
`forward::bubble_dynamics::gas_transfer::DiffusiveBubble`. It couples
Rayleigh–Plesset wall motion to a retained dissolved-gas profile and returns the
ambient equilibrium radius for the current gas inventory. Supply the absolute
far-field pressure at each step midpoint, including the ambient pressure during
quiet intervals. The [coupling design and verification](src/forward/bubble_dynamics/gas_transfer/DESIGN.md)
specify the physical inputs and benchmark range.

### Spectral Westervelt pressure update

The spectral Westervelt and Kuznetsov solvers apply the periodic Laplacian
with the same real-input FFT operator. Apollo owns the cached 3D plan and
executes both transforms through `kwavers_math::fft`. Between the forward
and inverse transforms, Kwavers multiplies each retained bin by `−|k|²`.
The optional `pinn` feature uses Coeus for automatic differentiation when
training inverse models.

For a grid of `nx × ny × nz` cells,
the operator retains `nx × ny × (nz / 2 + 1)` complex bins (integer division)
and three one-dimensional wavenumber tables. Its inverse writes directly to
the solver's real Laplacian buffer. Callers with their own full-spectrum
multipliers can use `compute_laplacian_spectral_into`.

The allocation regression advances a Fourier mode through the public pressure
update on both even and odd grids. The benchmark measures a warmed update
with a water medium and a nonzero pressure field.

```sh
cargo test --locked -p kwavers-solver --test allocation_contracts spectral_westervelt_warm
cargo bench --locked -p kwavers-solver --bench westervelt_step -- westervelt_spectral_step
```

On macOS arm64 with Rust 1.97.0, the same Criterion harness measured one
complete warm step at baseline `daf3842e` and implementation `63fff824`.
Each binary used the locked release profile, a one-second warm-up, and 20
samples per grid. Setup was outside the timed interval.

| Grid | Baseline median | Updated median | Reduction |
| --- | ---: | ---: | ---: |
| 32 × 24 × 32 | 0.993 ms | 0.824 ms | 17.1% |
| 32 × 24 × 45 | 1.471 ms | 1.147 ms | 22.1% |

On an 8 × 6 × 10 grid the calling-thread allocator recorded four allocations
and 8,000 bytes on the first warm baseline step. The updated regression
recorded zero allocations and reallocations on each of four warm steps, on
that grid and on a 9 × 6 × 7 grid. The regression also checked the pressure against
the discrete Fourier-mode recurrence on both grids. The measured configuration used a
homogeneous medium, a null source, and disabled conservation diagnostics.

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
