# kwavers-transducer

High-level transducer devices for kwavers: focused bowls, phased/linear/matrix/2-D/hemispherical arrays, k-Wave arrays, calibration, factories, beamforming, PAM, ultrafast.

`kwavers-transducer` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-source` / `kwavers-receiver`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-transducer = "4.1.0"
```

## Public modules

`array_2d`, `basic`, `beamforming`, `bulk_piezo`, `curvilinear`, `design`, `factory`, `flexible`, `hemispherical`, `kwave_array`, `mems`, `passive_acoustic_mapping`, `transducers`, `ultrafast`

## Documentation

- API reference: <https://docs.rs/kwavers-transducer>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
