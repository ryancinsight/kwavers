# Beamforming Validation Contract

`examples/beamforming_results.rs` is now a kwavers-backed validation example, not a standalone
near-field renderer.

Contract:

- Input is a generated full-stack v2 `DriverManifest` (`TX_0..TX_95`, four tile profiles, no legacy
  single-stim block), normally `output/manifests/v2_per_tile_stim.kv` or
  `output/full_driver/driver_manifest.kv`.
- The example builds `EnergyBudgetReport`, then calls `run_experiment(..., &KwaversSim, ...)`.
- `KwaversSim` routes the realized channel geometry through
  `kwavers-transducer::propagate_focused_linear_array`; the adapter converts the driver manifest's
  first-to-last channel-centre aperture span into kwavers-transducer's pitch-cell aperture span
  before propagation.
- The validation gate is `ExperimentRecord::beam_report`: focal pressure, mechanical index,
  grating-lobe-free geometry, and per-tile resistor margin must all pass before artifacts are
  emitted.
- `beamforming_metrics.csv`, `beamforming_validation.kv`, and `tile_geometry.csv` are the
  authoritative outputs for article-replication work. The BMP files are deterministic visualizations
  of the validated metrics and realized channel geometry; they are not the acoustic oracle.
- Regenerated 2026-06-29 values from `output/manifests/v2_per_tile_stim.kv`: 96 channels,
  11.027 MPa focal pressure, MI 7.797, 4108 W/cm2 ISPPA, 0.500 mm lateral 6 dB width,
  2.074 mm axial 6 dB width, grating-lobe-free true, far-field false, all checks pass.

Run:

```text
cargo run --release -p kwavers-driver --example v2_per_tile_stim
cargo run --release -p kwavers-driver --features kwavers --example beamforming_results -- output/beamforming output/manifests/v2_per_tile_stim.kv
```
