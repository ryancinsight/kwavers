# kwavers-receiver

Low-level acoustic recording primitives for
[kwavers](https://github.com/ryancinsight/kwavers): sensor-array geometry, field
recorders, point sensors, grid sampling, and sonoluminescence detection.

This crate is the single source of truth for *where* a simulation is observed. Sensor
positions defined here are what beamformers, passive acoustic maps, and reconstruction
pipelines downstream compute against.

Higher-level acquisition — beamforming, passive acoustic mapping, ultrafast sequences —
lives in [`kwavers-transducer`](https://docs.rs/kwavers-transducer), which depends on this
crate.

## What it provides

| Module | Responsibility |
|---|---|
| `array` | `SensorArray`, `Sensor`, `Position`, `SensorArrayGeometry` — array geometry SSOT |
| `recorder` | Pressure, velocity, and complex-field recorders over a time loop |
| `point` | `PointSensor`, `PointSensorConfig` — arbitrary off-grid sampling points |
| `grid_sampling` | `GridSensorSet`, `GridPoint` — sampling aligned to grid cells |
| `sonoluminescence` | Photon-emission detection for cavitation studies |

A `Sensor` carries its sensitivity (V/Pa) and directivity alongside its position, so an
array models the instrument rather than only its geometry.

## Example

```rust
use kwavers_receiver::{Position, Sensor, SensorArray, SensorArrayGeometry};

// 8-element linear array on 0.3 mm pitch.
let sensors: Vec<Sensor> = (0..8)
    .map(|i| Sensor::new(i, Position::new(i as f64 * 3e-4, 0.0, 0.0)))
    .collect();

let array = SensorArray::new(sensors, 1540.0, SensorArrayGeometry::Linear);

assert_eq!(array.num_sensors(), 8);
// The centroid sits half a pitch short of the last element.
assert!((array.centroid().x - 10.5e-4).abs() < 1e-12);
```

## Documentation

- API reference: <https://docs.rs/kwavers-receiver>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
