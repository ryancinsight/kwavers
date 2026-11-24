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
criterion_group!(benches, westervelt_step);
criterion_main!(benches);
