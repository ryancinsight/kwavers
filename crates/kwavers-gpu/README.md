# kwavers-gpu

Hephaestus-backed provider-generic GPU compute backend for kwavers.

`kwavers-gpu` is a member of the [kwavers](https://github.com/ryancinsight/kwavers) workspace, an interdisciplinary ultrasound-light physics simulation library. It sits in the GPU backend, above `kwavers-analysis`. The workspace is split into per-layer crates with strictly unidirectional dependencies and no facade: depend on the crates you need directly.

## Usage

```toml
[dependencies]
kwavers-gpu = "5.0.0"
```

## Public modules

`profiling`, `validation` — always available.

`backend`, `beamforming`, `gpu`, `pstd_gpu` — behind the `gpu` feature, which
pulls in the concrete provider implementations.

## Documentation

- API reference: <https://docs.rs/kwavers-gpu>
- Domain book: <https://ryancinsight.github.io/kwavers/>
- Crate map, layering rules, and quick start: [workspace README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT OR Apache-2.0 — see [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).
