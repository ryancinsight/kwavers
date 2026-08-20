# kwavers-imaging

Medical-imaging domain workflows for [kwavers](https://github.com/ryancinsight/kwavers):
DICOM/CT/NIfTI loaders, ultrasound and photoacoustic modality models, contrast-enhanced
ultrasound orchestration, and multimodality fusion.

This crate holds the imaging *domain* — the data structures and orchestration traits that
describe an imaging study — while the physics and solver layers supply the numerics that
implement them. Clinical code therefore depends on imaging abstractions rather than on
solver internals.

## Layers

1. **Domain models** (`ultrasound`, `photoacoustic`) — data structures for imaging concepts
2. **Orchestration interfaces** (`ceus_orchestrator`) — traits that physics and simulation
   layers implement
3. **Loaders and fusion** (`medical`, `unified_loader`, `fusion`, `multimodality_fusion`)

## What it provides

| Item | Responsibility |
|---|---|
| `MedicalImageLoader`, `DicomImageLoader`, `CTImageLoader`, `create_loader` | Format-specific ingestion behind one trait |
| `UnifiedMedicalImageLoader`, `MedicalImageBatchLoader` | Format-detecting and batched loading |
| `MedicalImageMetadata` | Spacing, orientation, and modality metadata carried with the volume |
| `CEUSOrchestrator`, `CEUSOrchestrators` | Contrast-enhanced ultrasound sequence orchestration |
| `FusionEngine`, `MultimodalityFusionManager`, `FusionConfig` | Registration and fusion across modalities |
| `RegistrationTransform`, `AffineTransform`, `TransformationType` | Spatial transforms between studies |
| `ImageData`, `ImageModality` | The modality-tagged image payload |

Image I/O and spatial-transform machinery are delegated to `ritk` (`ritk-io`,
`ritk-image`, `ritk-spatial`), the Atlas imaging toolkit; this crate owns the modality
semantics around them.

## Related crates

- Reconstruction pipelines and clinical diagnostic workflows —
  [`kwavers-diagnostics`](https://docs.rs/kwavers-diagnostics)
- Sensor and source primitives — [`kwavers-receiver`](https://docs.rs/kwavers-receiver),
  [`kwavers-source`](https://docs.rs/kwavers-source)
- Wave-equation specifications — [`kwavers-physics`](https://docs.rs/kwavers-physics)

## Documentation

- API reference: <https://docs.rs/kwavers-imaging>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
