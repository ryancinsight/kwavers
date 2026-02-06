// benches/grid_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::Grid;

fn grid_dimensions_benchmark(c: &mut Criterion) {
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4)
        .expect("Grid creation should succeed for valid parameters");

    c.bench_function("grid_dimensions", |b| {
        b.iter(|| {
            let dims = grid.dimensions();
            black_box(dims)
        })
    });
}

fn grid_spacing_benchmark(c: &mut Criterion) {
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4)
        .expect("Grid creation should succeed for valid parameters");

    c.bench_function("grid_spacing", |b| {
        b.iter(|| {
            let spacing = grid.spacing();
            black_box(spacing)
        })
    });
}

criterion_group!(benches, grid_dimensions_benchmark, grid_spacing_benchmark);
criterion_main!(benches);
