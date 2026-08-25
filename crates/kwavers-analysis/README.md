# kwavers-analysis

Analysis layer for [kwavers](https://github.com/ryancinsight/kwavers): signal processing,
beamforming support, validation, ML and uncertainty quantification, performance
measurement, and plotting/visualization.

Everything here runs *after* a field exists. The crate takes simulated or recorded data
and turns it into a result a study can report: a processed signal, a validated conservation
budget, an uncertainty interval, a performance profile, or a figure.

## What it provides

| Module | Responsibility |
|---|---|
| `signal_processing` | Filtering, envelope detection, spectral analysis of recorded traces |
| `conservation` | Mass/momentum/energy budgets over a run — the drift check for long integrations |
| `validation` | Convergence studies, analytical and benchmark comparisons |
| `ml` | Learned surrogates and uncertainty quantification |
| `performance` | Timing, throughput, and roofline-style measurement harnesses |
| `plotting` | Deterministic figure generation from result data |
| `distributed` | Multi-threaded analysis pipeline scheduling |
| `testing` | Shared harness utilities for value-semantic result assertions |

Figures are generated from the data by committed plotting code, so a chapter figure or a
validation plot can always be regenerated from the run that produced it.

## Features

- `gpu-visualization` — enables provider-neutral visualization transfer. The
  application selects Leto host storage or the Hephaestus GPU provider; the
  analysis crate does not acquire devices or depend on WGPU. Backend selection
  is intentionally absent from `VisualizationConfig`; callers select once at
  the top-level Kwavers composition boundary and inject the returned provider.

## Documentation

- API reference: <https://docs.rs/kwavers-analysis>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
