# kwavers-field

Numerical field definitions for kwavers: component indices (SSOT), field-type mapping, operations/statistics, and bubble/electromagnetic field states.

`kwavers-field` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-grid`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-field = "3.0.0"
```

## Public modules

`bubble`, `electromagnetic`, `indices`, `mapping`, `operations`, `r`, `wave`

## Documentation

- API reference: <https://docs.rs/kwavers-field>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
