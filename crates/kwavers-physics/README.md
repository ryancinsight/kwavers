# kwavers-physics

Physics for kwavers: nonlinear acoustics, bubble dynamics, thermal, optics, chemistry, elastic waves.

`kwavers-physics` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the physics layer, above the domain crates. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-physics = "3.0.0"
```

## Public modules

`acoustics`, `analytical`, `chemistry`, `electromagnetic`, `factory`, `field_surrogate`, `foundations`, `optics`, `photoacoustics`, `therapy`, `thermal`, `traits`

## Documentation

- API reference: <https://docs.rs/kwavers-physics>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
