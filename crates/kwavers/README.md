# kwavers

Top-level application and integration crate for the
[kwavers](https://github.com/ryancinsight/kwavers) ultrasound–light simulation workspace.

**This crate is not a facade.** It re-exports nothing. It carries the `kwavers` binary, the
cross-cutting integration tests, examples, and benchmarks, and a small set of application
utilities. Library consumers depend on the layer crates directly.

## Which crate do I depend on?

| Need | Crate |
|---|---|
| Errors, constants, arenas | [`kwavers-core`](https://docs.rs/kwavers-core) |
| FFT, linear algebra, numerics, SIMD | [`kwavers-math`](https://docs.rs/kwavers-math) |
| Grids, coordinates, geometric domains | [`kwavers-grid`](https://docs.rs/kwavers-grid) |
| Field component layout and operations | [`kwavers-field`](https://docs.rs/kwavers-field) |
| Excitation signals | [`kwavers-signal`](https://docs.rs/kwavers-signal) |
| Materials and tissue models | [`kwavers-medium`](https://docs.rs/kwavers-medium) |
| Tissue phantoms | [`kwavers-phantom`](https://docs.rs/kwavers-phantom) |
| Tetrahedral meshes (FEM/BEM) | [`kwavers-mesh`](https://docs.rs/kwavers-mesh) |
| Absorbing and variational boundaries | [`kwavers-boundary`](https://docs.rs/kwavers-boundary) |
| Source primitives | [`kwavers-source`](https://docs.rs/kwavers-source) |
| Sensor primitives | [`kwavers-receiver`](https://docs.rs/kwavers-receiver) |
| Transducer devices, beamforming, PAM | [`kwavers-transducer`](https://docs.rs/kwavers-transducer) |
| Imaging domain models and loaders | [`kwavers-imaging`](https://docs.rs/kwavers-imaging) |
| Governing equations | [`kwavers-physics`](https://docs.rs/kwavers-physics) |
| Forward and inverse solvers | [`kwavers-solver`](https://docs.rs/kwavers-solver) |
| GPU compute backends | [`kwavers-gpu`](https://docs.rs/kwavers-gpu) |
| Run orchestration and result I/O | [`kwavers-simulation`](https://docs.rs/kwavers-simulation) |
| Post-run analysis and validation | [`kwavers-analysis`](https://docs.rs/kwavers-analysis) |
| Diagnostic imaging workflows | [`kwavers-diagnostics`](https://docs.rs/kwavers-diagnostics) |
| Therapy planning, safety, regulatory | [`kwavers-therapy`](https://docs.rs/kwavers-therapy) |
| Python bindings | `pykwavers` (built from `kwavers-python`) |

## What is in this crate

| Item | Responsibility |
|---|---|
| `main.rs` | The `kwavers` binary |
| `theranostic` | Cross-layer theranostic feedback utilities that span diagnostics and therapy |
| `init_logging`, `get_version_info` | Application startup helpers |
| `tests/`, `examples/`, `benches/` | Cross-cutting integration coverage that no single layer crate owns |

## Example

```rust
let info = kwavers::get_version_info();
assert_eq!(info["name"], "kwavers");
assert!(info.contains_key("version"));
```

## Documentation

- API reference: <https://docs.rs/kwavers>
- Domain book, workspace layout, and development status:
  [repository README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
