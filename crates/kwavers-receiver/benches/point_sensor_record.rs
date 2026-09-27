//! Per-step cost of `PointSensor::record`: interpolation at every sensor plus
//! appending the row to the trace storage.
//!
//! 64 sensors on a 64^3 grid (field 2 MiB, larger than L2 on the reference
//! host) over 2000 steps; each iteration builds a fresh sensor so the trace
//! storage grows from empty exactly as in a simulation run.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers_grid::Grid;
use kwavers_receiver::{PointSensor, PointSensorConfig};
use leto::Array3;

const N: usize = 64;
const SENSORS: usize = 64;
const STEPS: usize = 2000;

fn record_steps(c: &mut Criterion) {
    let grid = Grid::new(N, N, N, 1e-3, 1e-3, 1e-3).expect("invariant: positive grid");
    let field = Array3::<f64>::from_shape_fn((N, N, N), |[i, j, k]| {
        (i as f64).sin() + (j as f64).cos() * (k as f64 + 1.0).ln()
    });
    let locations: Vec<[f64; 3]> = (0..SENSORS)
        .map(|s| {
            let t = s as f64 / SENSORS as f64;
            [0.005 + 0.05 * t, 0.030 + 0.001 * t, 0.010 + 0.04 * t]
        })
        .collect();

    c.bench_function("point_sensor_record_64x2000", |b| {
        b.iter(|| {
            let config = PointSensorConfig::new(black_box(locations.clone()));
            let mut sensor = PointSensor::new(config, &grid).expect("invariant: in-grid sensors");
            for step in 0..STEPS {
                sensor.record(field.view(), &grid, step);
            }
            black_box(sensor.max_pressure(SENSORS - 1))
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = record_steps
}
criterion_main!(benches);
