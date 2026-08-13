# kwavers-source

Low-level acoustic/optical/EM excitation primitives for kwavers: source trait, grid/mask sources, wavefronts, custom signals, apodization windows.

`kwavers-source` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-medium`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-source = "3.0.0"
```

## Public modules

`apodization`, `config`, `custom`, `electromagnetic`, `grid_source`, `injection`, `optical`, `structs`, `types`, `wavefront`

## Documentation

- API reference: <https://docs.rs/kwavers-source>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
