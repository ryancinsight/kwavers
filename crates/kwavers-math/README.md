# kwavers-math

Mathematical primitives for [kwavers](https://github.com/ryancinsight/kwavers): FFT,
linear algebra, numerics, geometry, statistics, and SIMD.

This crate sits directly above `kwavers-core` and below every domain layer. It has no
domain-specific dependencies — nothing here knows about tissue, transducers, or therapy —
so solvers, physics, and analysis can share one set of numerical building blocks.

## What it provides

| Module | Responsibility |
|---|---|
| `fft` | Apollo-backed FFT plans and caches, k-space utilities, spectral helpers |
| `geometry` | Geometric primitives and spatial computations |
| `linear_algebra` | Norms, decompositions, sparse matrices, matrix-free LSQR |
| `inverse_problems` | Regularization and inverse-problem solvers |
| `numerics` | Numerical operators, interpolation, integration |
| `apodization` | Aperture window shapes shared by source and beamforming layers |
| `simd_safe` | Safe wrappers over the `hermes-simd` kernels |

## Single source of truth

kwavers owns no separate FFT engine or dense-linear-algebra kernel. The canonical
implementations live in the Atlas first-party crates and are re-exported here so that
consumers have one import path and one implementation:

- FFT plans, caches, and shifts — `apollo`
- Optimization (`minimize`, L-BFGS), windows (`hann`, `hamming`, `blackman`, `tukey`),
  statistics (`pearson`, `rmse`, `psnr`, …), special functions (`erf`, `j0`, `j1`, `jn`,
  `sinc`), eigen/LSQR/sparse — `leto-ops`
- Complex scalars (`Complex32`, `Complex64`) — `eunomia`

The `Fft1d`/`Fft2d`/`Fft3d` aliases bind Apollo's scalar-generic plans to `f64`, the
precision the kwavers spectral layer works in end to end.

## Example

A real-to-complex round trip through the FFT facade:

```rust
use kwavers_math::fft::{fft_1d_array, ifft_1d_array};
use leto::Array1;

let signal = Array1::from_vec(4, vec![1.0, 0.0, -1.0, 0.0]).unwrap();
let spectrum = fft_1d_array(&signal);
let recovered = ifft_1d_array(&spectrum);

for i in 0..4 {
    assert!((recovered[[i]] - signal[[i]]).abs() < 1e-12);
}
```

## Documentation

- API reference: <https://docs.rs/kwavers-math>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
