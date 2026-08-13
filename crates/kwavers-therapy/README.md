# kwavers-therapy

Clinical therapy + care-delivery for kwavers: HIFU/histotripsy/lithotripsy planning, theranostic guidance, dose & safety monitoring, regulatory, patient management.

`kwavers-therapy` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the clinical layer, above `kwavers-simulation`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-therapy = "3.0.0"
```

## Public modules

`patient_management`, `regulatory`, `safety`, `therapy`

## Documentation

- API reference: <https://docs.rs/kwavers-therapy>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
