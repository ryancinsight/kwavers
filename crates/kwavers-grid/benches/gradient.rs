//! End-to-end gradient timings include allocation of the three returned volumes.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kwavers_grid::{
    operators::{
        gradient, gradient_optimized::BoundaryStrategy, FdAccuracyOrder, GradientCache,
        GradientOperatorBuilder,
    },
    Grid,
};
use leto::Array3;
use std::{hint::black_box, time::Duration};

fn gradient_benchmark(c: &mut Criterion) {
    let grid = Grid::new(48, 48, 48, 0.25, 0.5, 2.0).unwrap();
    let mut field = Array3::<f64>::zeros([48, 48, 48]);
    for (i, x) in field.iter_mut().enumerate() {
        *x = f64::from(u16::try_from(i % 1009).unwrap());
    }
    let cache = GradientCache::new(&grid);
    let order = FdAccuracyOrder::Fourth;
    let mut group = c.benchmark_group("cartesian_gradient");
    group.throughput(Throughput::Elements(48_u64.pow(3)));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("interior", |b| {
        b.iter(|| gradient(black_box(&field.view()), &grid, order).unwrap())
    });
    for (parallel, name) in [(false, "periodic_serial"), (true, "periodic_parallel")] {
        let op = GradientOperatorBuilder::new()
            .with_parallel(parallel)
            .with_boundary_strategy(BoundaryStrategy::Periodic)
            .build();
        group.bench_function(name, |b| {
            b.iter(|| {
                op.compute(black_box(&field.view()), &grid, order, Some(&cache))
                    .unwrap()
            })
        });
    }
    group.finish();
}
criterion_group!(benches, gradient_benchmark);
criterion_main!(benches);
