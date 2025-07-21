// benches/validation_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::{Grid, HomogeneousMedium};

fn grid_validation_benchmark(c: &mut Criterion) {
    let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4).unwrap();
    
    c.bench_function("grid_validation", |b| {
        b.iter(|| {
            let result = grid.validate();
            black_box(result)
        })
    });
}

fn medium_validation_benchmark(c: &mut Criterion) {
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0).unwrap();
    
    c.bench_function("medium_validation", |b| {
        b.iter(|| {
            let result = medium.validate();
            black_box(result)
        })
    });
}

criterion_group!(benches, grid_validation_benchmark, medium_validation_benchmark);
criterion_main!(benches);