# kwavers-receiver

Low-level acoustic recording primitives for kwavers: sensor-array geometry, field recorders, point sensors, grid sampling, sonoluminescence detection.

`kwavers-receiver` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-field`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-receiver = "3.0.0"
```

## Public modules

`array`, `grid_sampling`, `point`, `recorder`, `sonoluminescence`

## Documentation

- API reference: <https://docs.rs/kwavers-receiver>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
