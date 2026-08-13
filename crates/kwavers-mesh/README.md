# kwavers-mesh

Tetrahedral mesh infrastructure for kwavers FEM: nodes, connectivity, statistics, quality metrics, gaia bridge.

`kwavers-mesh` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-grid`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-mesh = "3.0.0"
```

## Public modules

`tetrahedral`

## Documentation

- API reference: <https://docs.rs/kwavers-mesh>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
