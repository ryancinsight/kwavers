# kwavers-solver

Forward and inverse solvers for kwavers: FDTD/PSTD/k-space/Helmholtz, BEM, FWI/RTM/CBS, PINN.

`kwavers-solver` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the solver layer, above `kwavers-physics`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-solver = "3.0.0"
```

## Public modules

`analytical`, `backend`, `config`, `constants`, `factory`, `feature`, `forward`, `geometry`, `integration`, `interface`, `inverse`, `multiphysics`, `plugin`, `safety`, `utilities`, `validation`, `workspace`

## Documentation

- API reference: <https://docs.rs/kwavers-solver>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
