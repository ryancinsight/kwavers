# kwavers-signal

Excitation signal generation and processing for kwavers: waveforms, pulses, frequency sweeps, modulation, windowing, filters.

`kwavers-signal` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the domain layer, above `kwavers-math`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-signal = "3.0.0"
```

## Public modules

`amplitude`, `analytic`, `filter`, `frequency`, `frequency_sweep`, `functions`, `modulation`, `phase`, `pulse`, `special`, `traits`, `waveform`, `window`

## Documentation

- API reference: <https://docs.rs/kwavers-signal>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
