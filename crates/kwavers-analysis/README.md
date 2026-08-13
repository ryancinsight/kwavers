# kwavers-analysis

Analysis layer for kwavers: signal processing, beamforming, validation, ML/uncertainty, performance, plotting/visualization.

`kwavers-analysis` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the analysis layer, above `kwavers-solver`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-analysis = "3.0.0"
```

## Public modules

`conservation`, `distributed`, `ml`, `performance`, `plotting`, `signal_processing`, `testing`, `validation` — always available; `visualization` is behind the `gpu-visualization` feature.

## Documentation

- API reference: <https://docs.rs/kwavers-analysis>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
