# Appendix B — Atlas Crate Dependencies

## Dependency Graph (kwavers perspective)

```
eunomia          → scalar/complex traits (replaces num-traits/num-complex)
mnemosyne/themis → memory allocator stack
hermes           → SIMD abstraction (replaces direct intrinsics)
moirai           → async runtime + parallel execution (replaces tokio/rayon)
leto             → CPU arrays + geometry (replaces ndarray/nalgebra)
leto-ops         → linear algebra kernels on leto arrays (replaces nalgebra solvers)
hephaestus       → GPU arrays (leto-aligned API; replaces wgpu-based ad-hoc buffers)
coeus            → neural network + autodiff (replaces burn)
apollo           → FFT (replaces rustfft)
ritk             → image toolkit (consumes coeus + leto)
kwavers          → acoustic simulation (consumes all of the above)
```

## Version Pinning (workspace Cargo.toml pattern)

```toml
[workspace.dependencies]
eunomia   = { git = "https://github.com/ryancinsight/eunomia",   default-features = false, features = ["std"] }
leto      = { git = "https://github.com/ryancinsight/leto.git",  default-features = false, features = ["std"] }
leto-ops  = { git = "https://github.com/ryancinsight/leto.git",  default-features = false, features = ["std"] }
moirai    = { git = "https://github.com/ryancinsight/Moirai.git", default-features = false, features = ["parallel","mnemosyne","melinoe"] }
hermes-simd = { git = "https://github.com/ryancinsight/hermes.git", default-features = false, features = ["std"] }
apollo    = { package = "apollo-fft", path = "../apollo/crates/apollo-fft" }
coeus-core     = { path = "../coeus/coeus-core" }
coeus-autograd = { path = "../coeus/coeus-autograd" }
mnemosyne = { path = "../mnemosyne/crates/mnemosyne", default-features = false, features = ["std_tls","eunomia"] }
```

## Local Path Patches

```toml
[patch."https://github.com/ryancinsight/leto.git"]
leto     = { path = "../leto/crates/leto" }
leto-ops = { path = "../leto/crates/leto-ops" }

[patch."https://github.com/ryancinsight/eunomia"]
eunomia = { path = "../eunomia/crates/eunomia" }
```

## Atlas Repository Map

| Crate | Repo path |
|---|---|
| `eunomia` | `repos/eunomia` |
| `leto`, `leto-ops` | `repos/leto` |
| `hermes-simd` | `repos/hermes` |
| `moirai*` | `repos/moirai` |
| `mnemosyne*` | `repos/mnemosyne` |
| `themis` | `repos/themis` |
| `coeus-*` | `repos/coeus` |
| `apollo-fft` | `repos/apollo` |
| `hephaestus-*` | `repos/hephaestus` |
| `ritk-*` | `repos/ritk` |
| `kwavers-*` | `repos/kwavers` |
