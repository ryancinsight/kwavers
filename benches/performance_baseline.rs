//! Performance baseline benchmarks for Kwavers
//! 
//! These benchmarks establish performance baselines for critical operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kwavers::{Grid, Time};
use kwavers::medium::{
    core::CoreMedium,
    homogeneous::HomogeneousMedium,
};
use ndarray::Array3;

fn grid_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_creation");
    
    for size in [32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                Grid::new(size, size, size, 1e-3, 1e-3, 1e-3)
            });
        });
    }
    
    group.finish();
}

fn field_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_creation");
    
    for size in [32, 64, 128].iter() {
        let grid = Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3);
        group.bench_with_input(BenchmarkId::from_parameter(size), &grid, |b, grid| {
            b.iter(|| {
                grid.create_field()
            });
        });
    }
    
    group.finish();
}

fn field_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_operations");
    
    // Benchmark field addition
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let field1 = grid.create_field();
    let field2 = grid.create_field();
    
    group.bench_function("field_add_64", |b| {
        b.iter(|| {
            let result = &field1 + &field2;
            black_box(result);
        });
    });
    
    // Benchmark field multiplication
    group.bench_function("field_mul_64", |b| {
        b.iter(|| {
            let result = &field1 * 2.0;
            black_box(result);
        });
    });
    
    group.finish();
}

fn medium_evaluation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_evaluation");
    
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 1e-3, 0.072, &grid);
    
    use kwavers::medium::Medium;
    
    group.bench_function("density_lookup", |b| {
        b.iter(|| {
            medium.density(
                black_box(32e-3), 
                black_box(32e-3), 
                black_box(32e-3), 
                &grid
            )
        });
    });
    
    group.bench_function("sound_speed_lookup", |b| {
        b.iter(|| {
            medium.sound_speed(
                black_box(32e-3), 
                black_box(32e-3), 
                black_box(32e-3), 
                &grid
            )
        });
    });
    
    group.finish();
}

fn position_to_indices_benchmark(c: &mut Criterion) {
    let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);
    
    c.bench_function("position_to_indices", |b| {
        b.iter(|| {
            grid.position_to_indices(
                black_box(64e-3),
                black_box(64e-3),
                black_box(64e-3)
            )
        });
    });
}

fn memory_usage_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Measure memory allocation for different grid sizes
    for size in [32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::new("grid_memory", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let grid = Grid::new(size, size, size, 1e-3, 1e-3, 1e-3);
                    let _field = grid.create_field();
                    black_box(grid);
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    grid_creation_benchmark,
    field_creation_benchmark,
    field_operations_benchmark,
    medium_evaluation_benchmark,
    position_to_indices_benchmark,
    memory_usage_benchmark
);

criterion_main!(benches);