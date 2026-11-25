//! Four consecutive Westervelt updates, including history bootstrap and reuse.
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};
use leto::Array3;
use std::{hint::black_box, time::Duration};

fn westervelt_step(c: &mut Criterion) {
    let grid = Grid::new(48, 48, 48, 0.001, 0.0012, 0.0015).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let mut initial = Array3::zeros([48, 48, 48]);
    for (i, p) in initial.iter_mut().enumerate() {
        *p = f64::from(u16::try_from(i % 971).unwrap());
    }
    let dt = 1e-7;
    let mut group = c.benchmark_group("westervelt_four_steps");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(4 * 48_u64.pow(3)));
    for order in [2, 4, 6] {
        let config = WesterveltFdtdConfig {
            spatial_order: order,
            ..Default::default()
        };
        group.bench_function(format!("order_{order}"), |b| {
            b.iter_batched_ref(
                || {
                    let mut solver = WesterveltFdtd::new(config.clone(), &grid, &medium);
                    solver.set_pressure_field(initial.view());
                    solver
                },
                |solver| {
                    for n in 0..4 {
                        solver
                            .update(&medium, &grid, &[], f64::from(n) * dt, dt)
                            .unwrap();
                    }
                    black_box(solver.pressure());
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}
/// Measure one complete warm step from the same nonzero periodic pressure state.
fn spectral_step(c: &mut Criterion) {
    use kwavers_physics::traits::AcousticWaveModel;
    use kwavers_solver::forward::nonlinear::westervelt_spectral::WesterveltWave;
    use kwavers_source::NullSource;
    use leto::Array4;

    let mut group = c.benchmark_group("westervelt_spectral_step");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for nz in [32_u32, 45] {
        let shape = [32, 24, usize::try_from(nz).unwrap()];
        let [nx, ny, nz_grid] = shape;
        let grid = Grid::new(nx, ny, nz_grid, 1.0e-3, 1.2e-3, 1.5e-3).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let initial = Array3::from_shape_fn(shape, |[i, _, k]| {
            let phase = i as f64 / nx as f64 + k as f64 / f64::from(nz);
            (std::f64::consts::TAU * phase).cos()
        });
        let previous = Array3::zeros(shape);
        let source = NullSource::new();
        let dt = 1.0e-7;
        group.throughput(Throughput::Elements(32 * 24 * u64::from(nz)));
        group.bench_function(format!("32x24x{nz}"), |b| {
            b.iter_batched_ref(
                || {
                    let mut wave = WesterveltWave::new(&grid);
                    let mut fields = Array4::zeros([1, nx, ny, nz_grid]);
                    fields.index_axis_mut(0, 0).unwrap().assign(&initial);
                    wave.update_wave(&mut fields, &previous, &source, &grid, &medium, dt, 0.0)
                        .unwrap();
                    (wave, fields)
                },
                |(wave, fields)| {
                    wave.update_wave(fields, &previous, &source, &grid, &medium, dt, dt)
                        .unwrap();
                    black_box(fields);
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, westervelt_step, spectral_step);
criterion_main!(benches);
