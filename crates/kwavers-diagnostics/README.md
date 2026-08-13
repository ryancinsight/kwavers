# kwavers-diagnostics

Clinical diagnostic imaging workflows for kwavers: reconstruction pipelines, multi-modal fusion, Doppler, spectroscopy, functional ultrasound, decision support.

`kwavers-diagnostics` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the clinical layer, above `kwavers-simulation`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-diagnostics = "3.0.0"
```

## Public modules

`functional_ultrasound`, `photoacoustic`, `reconstruction`, `workflows`

## Documentation

- API reference: <https://docs.rs/kwavers-diagnostics>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
