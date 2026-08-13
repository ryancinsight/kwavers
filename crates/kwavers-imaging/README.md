# kwavers-imaging

Medical-imaging domain workflows for kwavers: DICOM/CT/NIfTI loaders, ultrasound/photoacoustic modalities, CEUS orchestration, multimodality fusion.

`kwavers-imaging` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-medium`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-imaging = "3.0.0"
```

## Public modules

`ceus_orchestrator`, `fusion`, `medical`, `multimodality_fusion`, `photoacoustic`, `ultrasound`, `unified_loader`

## Documentation

- API reference: <https://docs.rs/kwavers-imaging>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
