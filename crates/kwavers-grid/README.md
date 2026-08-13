# kwavers-grid

Spatial discretization for kwavers: Cartesian/cylindrical grids, coordinates, topology, operators, k-space FFT utilities, and geometric domains.

`kwavers-grid` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-math`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-grid = "3.0.0"
```

## Public modules

`adapter`, `config`, `coordinates`, `error`, `geometry`, `operators`, `simple_config`, `stability`, `structure`, `topology`, `validation`

## Documentation

- API reference: <https://docs.rs/kwavers-grid>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
