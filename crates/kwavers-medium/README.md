# kwavers-medium

Material and tissue models for kwavers: homogeneous/heterogeneous media, acoustic/elastic/optical/thermal/viscous properties, absorption models, anisotropy.

`kwavers-medium` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-grid`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-medium = "4.0.0"
```

## Public modules

`absorption`, `acoustic`, `adapters`, `analytical_properties`, `anisotropic`, `bubble`, `builder`, `config`, `core`, `elastic`, `error`, `frequency_dependent`, `heterogeneous`, `homogeneous`, `interface`, `iterators`, `material_fields`, `optical`, `optical_map`, `properties`, `thermal`, `traits`, `validation_simulation`, `viscoelastic`, `viscous`, `wrapper`

## Documentation

- API reference: <https://docs.rs/kwavers-medium>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
