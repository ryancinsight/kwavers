//! Benchmark one application to a preallocated 64³ volume.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kwavers_grid::{
    operators::{
        laplacian::{LaplacianBoundary, LaplacianConfig, LaplacianOperator},
        FdAccuracyOrder,
    },
    Grid,
};
use leto::Array3;
use std::{hint::black_box, time::Duration};

fn laplacian(c: &mut Criterion) {
    let grid = Grid::new(64, 64, 64, 0.5, 0.75, 1.0).unwrap();
    let mut input = Array3::zeros([64, 64, 64]);
    for (i, x) in input.iter_mut().enumerate() {
        *x = f64::from(u16::try_from(i % 20011).unwrap());
    }
    let mut output = Array3::zeros([64, 64, 64]);
    let mut group = c.benchmark_group("laplacian_apply_mut");
    group.throughput(Throughput::Elements(64_u64.pow(3)));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    for boundary in [
        LaplacianBoundary::Dirichlet,
        LaplacianBoundary::Neumann,
        LaplacianBoundary::Periodic,
    ] {
        for order in [
            FdAccuracyOrder::Second,
            FdAccuracyOrder::Fourth,
            FdAccuracyOrder::Sixth,
            FdAccuracyOrder::Eighth,
        ] {
            let op = LaplacianOperator::new(&grid, LaplacianConfig { order, boundary });
            group.bench_function(
                BenchmarkId::new(format!("{boundary:?}"), format!("{order:?}")),
                |b| {
                    b.iter(|| {
                        op.apply_mut(black_box(input.view()), output.view_mut())
                            .unwrap();
                        black_box(&output);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, laplacian);
criterion_main!(benches);
