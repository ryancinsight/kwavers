# kwavers-simulation

Simulation orchestration for kwavers: builders, runners, multi-physics coupling, modality pipelines, backends.

`kwavers-simulation` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the orchestration layer, above `kwavers-gpu`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-simulation = "3.0.0"
```

## Public modules

`backends`, `builder`, `configs`, `configuration`, `core`, `dispatch`, `factory`, `imaging`, `io`, `manager`, `modalities`, `multi_physics`, `parameters`, `runner`, `setup`, `solver_adapters`, `solver_factory`, `therapy`, `types`

## Documentation

- API reference: <https://docs.rs/kwavers-simulation>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
