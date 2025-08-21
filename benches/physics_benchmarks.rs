// benches/physics_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::{medium::homogeneous::HomogeneousMedium, Grid};

fn grid_creation_benchmark(c: &mut Criterion) {
    c.bench_function("grid_creation", |b| {
        b.iter(|| {
            let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
            black_box(grid)
        })
    });
}

fn medium_creation_benchmark(c: &mut Criterion) {
    c.bench_function("medium_creation", |b| {
        b.iter(|| {
            let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
            let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
            black_box(medium)
        })
    });
}

criterion_group!(benches, grid_creation_benchmark, medium_creation_benchmark);
criterion_main!(benches);
