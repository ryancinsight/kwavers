# Appendix B — Atlas Crate Dependency Map

Kwavers sits at the top of the Atlas dependency hierarchy, consuming
specialised crates from the ground up. The map below shows the primary
dependency relationships.

## Dependency Layers

```text
                         kwavers
                        /   |   \
          kwavers-python  kwavers-solver  kwavers-simulation
                              |               |
                      kwavers-physics    kwavers-core
                              |               |
                         kwavers-math    (domain types)
                         /    |    \
                      fft  linalg  signal
                       |      |       |
                    apollo  leto-ops  leto-ops
                               |
                             leto
                               |
                    eunomia  mnemosyne  hermes  moirai
```

## Atlas Crate Roles

| Crate | Replaces | Role in kwavers |
|-------|----------|-----------------|
| `leto` | `ndarray` + `nalgebra` | Arrays, geometry, linear algebra |
| `leto-ops` | Various | Numerical algorithms SSOT |
| `eunomia` | `num-traits` + `num-complex` | Scalar traits |
| `hermes` | x86 intrinsics | SIMD abstraction |
| `moirai` | `tokio` + `rayon` | Concurrency |
| `mnemosyne` | system allocator | Arena allocation |
| `themis` | — | Allocation strategies |
| `apollo` | `rustfft` | FFT (forward + autodiff) |
| `coeus` | `burn` | ML framework (PINNs) |
| `coeus-autograd` | — | Autodiff tape |
| `hephaestus` | GPU backends | GPU arrays |
| `ritk` | ITK | Image I/O, registration |
| `melinoe` | newtype macros | Branded domain types |

## Key Design Invariants

- **No `nalgebra`**: All linear algebra through `leto::geometry` and `leto-ops::linalg`.
- **No `ndarray`**: All array storage through `leto::Array*` and `leto::ArrayView*`.
- **No `burn`**: All ML through `coeus` + `coeus-autograd`.
- **No `tokio`/`rayon`**: All parallelism through `moirai`.
- **No `rustfft`**: All FFT through `apollo-fft`.
- **Python boundary only**: `numpy::ndarray` appears only in `kwavers-python` bindings
  (required for PyO3 array exchange — intentional, not a migration gap).

## Crate Location

All Atlas crates live under `D:\atlas\repos\`:

| Crate | Repo |
|-------|------|
| `leto`, `leto-ops` | `repos/leto` |
| `eunomia` | `repos/eunomia` |
| `hermes` | `repos/hermes` |
| `moirai` | `repos/moirai` |
| `mnemosyne`, `themis` | `repos/mnemosyne`, `repos/themis` |
| `apollo`, `apollo-fft` | `repos/apollo` |
| `coeus`, `coeus-autograd`, `coeus-nn` | `repos/coeus` |
| `hephaestus` | `repos/hephaestus` |
| `ritk` | `repos/ritk` |
| `melinoe` | `repos/melinoe` |
| `gaia` | `repos/gaia` |
