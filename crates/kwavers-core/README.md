# kwavers-core

Foundation layer for kwavers: constants, error types, arena allocation, time and logging utilities.

`kwavers-core` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the foundation layer (no first-party dependencies). The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-core = "3.0.0"
```

## Public modules

`arena`, `constants`, `error`, `log`, `time`, `units`, `utils`

## Documentation

- API reference: <https://docs.rs/kwavers-core>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
