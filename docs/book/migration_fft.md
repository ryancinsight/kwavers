# Chapter 41 — FFT: Apollo for Spectral Methods

`apollo-fft` replaces `rustfft` as the Atlas FFT engine. The kwavers spectral
solvers (PSTD, k-space correction, KZK) all route through apollo.

## What Changed

| Before (rustfft) | After (apollo) |
|---|---|
| `FftPlanner::new().plan_fft_forward(n)` | `FftPlan3D::<f64>::new(Shape3D{nx,ny,nz})` |
| `planner.process(&mut buffer)` | `plan.forward(&input, &mut output)` |
| `Arc::clone(&plan)` | `PlanCacheProvider::get_3d_plan(shape)` (reused automatically) |
| No autodiff | `apollo` integrates with `coeus-autograd` for gradient-through-FFT |

## kwavers facade

`kwavers-math::fft` wraps apollo with type aliases so existing kwavers code
uses `Fft3d` without changing call sites:

```rust
// kwavers-math/src/fft/mod.rs
pub use apollo::{FftPlan3D, PlanCacheProvider, Shape3D, Normalization};
pub type Fft3d = FftPlan3D<f64>;

pub fn get_fft_for_grid(nx: usize, ny: usize, nz: usize) -> Arc<Fft3d> {
    <f64 as PlanCacheProvider>::get_3d_plan(Shape3D { nx, ny, nz })
}
```

## Example: PSTD pressure step

```rust
use kwavers_math::fft::{get_fft_for_grid, Fft3d};
use apollo::Normalization;

let plan: Arc<Fft3d> = get_fft_for_grid(nx, ny, nz);

// Forward transform
let mut spectrum = vec![eunomia::Complex64::default(); nx * ny * nz];
plan.forward(&pressure_field, &mut spectrum, Normalization::None);

// Apply k-space filter in frequency domain …

// Inverse transform (in-place)
plan.inverse(&spectrum, &mut pressure_field, Normalization::Unitary);
```

## Plan Caching

Apollo maintains a thread-local + global LRU plan cache keyed by (shape,
scalar type). `get_fft_for_grid` resolves the cache on every call — no
global mutation, no mutex contention, safe for parallel solver steps.

## Autodiff Integration

Apollo FFT plans are `coeus-autograd`-aware: wrapping a plan call inside a
`Var` forward pass allows gradients to flow through the FFT for PINN and
FWI learned-physics objectives.
