// benches/validation_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::{medium::homogeneous::HomogeneousMedium, Grid, Medium};

fn system_validation_benchmark(c: &mut Criterion) {
    c.bench_function("validation_pipeline", |b| {
        b.iter(|| {
            // System validation - just test basic grid creation
            let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4);
            black_box(grid)
        })
    });
}

fn medium_validation_benchmark(c: &mut Criterion) {
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, &grid);

    c.bench_function("medium_validation", |b| {
        b.iter(|| {
            let density = medium.density(0.0, 0.0, 0.0, &grid);
            black_box(density)
        })
    });
}

criterion_group!(
    benches,
    system_validation_benchmark,
    medium_validation_benchmark
);
criterion_main!(benches);
