# kwavers

Top-level application and integration crate for the
[kwavers](https://github.com/ryancinsight/kwavers) ultrasound–light simulation workspace.

This crate carries the `kwavers` binary, cross-cutting integration tests, examples, and
benchmarks, plus a small set of application utilities. Library consumers generally depend
on the layer crates directly; the `gpu-visualization` feature also re-exports the explicit
visualization backend selector for application-bound setup.

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
| Selectable visualization transfer | `kwavers::visualization::VisualizationBackend` with `gpu-visualization` |
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

With `gpu-visualization`, application setup selects the provider explicitly and
injects it into the analysis engine. `Hephaestus` acquires the real GPU device;
`Leto` keeps the transfer on the host path.

```rust,ignore
use kwavers::visualization::{create_visualization_provider, VisualizationBackend};
use kwavers_analysis::visualization::{TransferMode, VisualizationConfig, VisualizationEngine};

async fn configure() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = VisualizationEngine::create(VisualizationConfig::default())?;
    let provider = create_visualization_provider(VisualizationBackend::Hephaestus)?;
    engine.set_transfer_provider(provider);
    engine.set_transfer_mode(TransferMode::Streaming);
    engine.initialize_gpu().await?;
    Ok(())
}
```

Replace `VisualizationBackend::Hephaestus` with
`VisualizationBackend::Leto` to select the host provider. A failed Hephaestus
acquisition is returned to the caller; it never silently selects Leto.

## Documentation

- API reference: <https://docs.rs/kwavers>
- Domain book, workspace layout, and development status:
  [repository README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
