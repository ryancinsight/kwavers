# kwavers-transducer

High-level transducer devices for [kwavers](https://github.com/ryancinsight/kwavers):
focused bowls, phased/linear/matrix/2-D/hemispherical arrays, k-Wave array compatibility,
calibration, source factories, beamforming, passive acoustic mapping, and ultrafast
acquisition.

This is the device layer. It composes the excitation primitives of
[`kwavers-source`](https://docs.rs/kwavers-source) and the recording primitives of
[`kwavers-receiver`](https://docs.rs/kwavers-receiver) into the physical instruments a
study is actually run with — and the acquisition stacks that depend on their geometry.

## Device families

| Module | Devices |
|---|---|
| `basic` | `LinearArray`, `MatrixArray`, `PistonSource` |
| `transducers::focused` | Bowls, spherical caps, arcs, annular and multi-bowl arrays |
| `array_2d` | `TransducerArray2D` with curvature and per-element apodization |
| `curvilinear` | Convex/curved diagnostic arrays |
| `hemispherical` | `HemisphericalArray` with element state, steering, and sparse-array optimization |
| `flexible` | `FlexibleTransducerArray` plus `CalibrationManager` for measured geometry |
| `kwave_array` | `KWaveArray` — k-Wave off-grid source compatibility |
| `bulk_piezo` / `mems` | Element-level physical models |

## Acquisition stacks

| Module | Responsibility |
|---|---|
| `beamforming` | Delay-and-sum and adaptive beamformers over a `SensorArray` |
| `passive_acoustic_mapping` | PAM for cavitation localization |
| `ultrafast` | Plane-wave and diverging-wave sequence acquisition |
| `design` | Aperture design specs, array synthesis, focused-array propagation |
| `factory` | `SourceFactory` — configuration-driven device construction |

## Example

```rust
use std::sync::Arc;
use kwavers_signal::SineWave;
use kwavers_source::{apodization::HanningApodization, Source};
use kwavers_transducer::LinearArray;
use kwavers_grid::Grid;

let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4).unwrap();
let signal = Arc::new(SineWave::new(2.0e6, 1.0e6, 0.0));

// 64-element, 19.2 mm aperture at 2 MHz in soft tissue, Hanning-apodized.
let array = LinearArray::new(
    19.2e-3,
    64,
    (3.2e-3, 3.2e-3, 0.0),
    signal,
    1540.0,
    2.0e6,
    HanningApodization,
);

let mask = array.create_mask(&grid);
// The aperture occupies grid cells.
assert!(mask.iter().any(|&w| w > 0.0));
```

## Documentation

- API reference: <https://docs.rs/kwavers-transducer>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
