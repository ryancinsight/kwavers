# kwavers-math

Mathematical primitives for kwavers: FFT, linear algebra, numerics, geometry, statistics, SIMD.

`kwavers-math` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the math layer, above `kwavers-core`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-math = "3.0.0"
```

## Public modules

`apodization`, `fft`, `geometry`, `inverse_problems`, `linear_algebra`, `numerics`, `simd`, `simd_safe`

## Documentation

- API reference: <https://docs.rs/kwavers-math>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
