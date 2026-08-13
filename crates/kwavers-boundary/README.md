# kwavers-boundary

Boundary conditions for kwavers: CPML/PML absorbing layers, FEM/BEM, multiphysics coupling, periodic boundaries, smoothing.

`kwavers-boundary` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-medium`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-boundary = "3.0.0"
```

## Public modules

`bem`, `config`, `coupling`, `cpml`, `fem`, `field_updater`, `periodic`, `pml`, `smoothing`, `traits`, `types`

## Documentation

- API reference: <https://docs.rs/kwavers-boundary>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
