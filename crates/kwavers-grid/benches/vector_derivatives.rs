//! Public, allocating entry points on a deterministic 48³ vector field.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kwavers_grid::{
    operators::{curl, divergence, FdAccuracyOrder},
    Grid,
};
use leto::Array3;
use std::{hint::black_box, time::Duration};

fn vector_derivatives(c: &mut Criterion) {
    let grid = Grid::new(48, 48, 48, 0.25, 0.5, 2.0).unwrap();
    let [x, y, z] = [17_usize, 31, 61].map(|factor| {
        let mut field = Array3::<f64>::zeros([48, 48, 48]);
        for (i, x) in field.iter_mut().enumerate() {
            *x = f64::from(u16::try_from(i * factor % 1009).unwrap());
        }
        field
    });
    let [x, y, z] = [x.view(), y.view(), z.view()];
    let mut group = c.benchmark_group("vector_derivatives");
    group.throughput(Throughput::Elements(48_u64.pow(3)));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_secs(1));
    for order in [FdAccuracyOrder::Second, FdAccuracyOrder::Eighth] {
        group.bench_function(format!("curl_{order:?}"), |b| {
            b.iter(|| curl(black_box(&x), &y, &z, &grid, order).unwrap())
        });
        group.bench_function(format!("divergence_{order:?}"), |b| {
            b.iter(|| divergence(black_box(&x), &y, &z, &grid, order).unwrap())
        });
    }
    group.finish();
}
criterion_group!(benches, vector_derivatives);
criterion_main!(benches);
