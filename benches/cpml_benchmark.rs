//! Benchmark comparing C-PML and standard PML performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kwavers::boundary::{CPMLBoundary, CPMLConfig, PMLBoundary, PMLConfig, Boundary};
use kwavers::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

fn create_test_field(grid: &Grid, angle: f64) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let angle_rad = angle * PI / 180.0;
    let k = 2.0 * PI * 1e6 / 1500.0;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k_idx in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let phase = k * (x * angle_rad.cos() + y * angle_rad.sin());
                field[[i, j, k_idx]] = phase.sin();
            }
        }
    }
    
    field
}

fn bench_pml_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("PML Comparison");
    
    // Test different grid sizes
    let grid_sizes = vec![64, 128, 256];
    
    for size in grid_sizes {
        let grid = Grid::new(size, size, size, 1e-3, 1e-3, 1e-3);
        
        // Standard PML
        let pml_config = PMLConfig::default();
        let mut standard_pml = PMLBoundary::new(pml_config).unwrap();
        
        // C-PML
        let cpml_config = CPMLConfig::default();
        let mut cpml = CPMLBoundary::new(cpml_config, &grid).unwrap();
        
        // Test field
        let mut field = create_test_field(&grid, 45.0);
        
        group.bench_with_input(
            BenchmarkId::new("Standard PML", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut field_copy = field.clone();
                    standard_pml.apply_acoustic(&mut field_copy, &grid, 0).unwrap();
                    black_box(field_copy);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("C-PML", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut field_copy = field.clone();
                    cpml.apply_acoustic(&mut field_copy, &grid, 0).unwrap();
                    black_box(field_copy);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_grazing_angles(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grazing Angle Performance");
    
    let grid = Grid::new(200, 200, 100, 1e-3, 1e-3, 1e-3);
    let angles = vec![0.0, 45.0, 75.0, 85.0];
    
    for angle in angles {
        // Standard PML
        let pml_config = PMLConfig::default();
        let mut standard_pml = PMLBoundary::new(pml_config).unwrap();
        
        // C-PML optimized for grazing
        let cpml_config = CPMLConfig::for_grazing_angles();
        let mut cpml_grazing = CPMLBoundary::new(cpml_config, &grid).unwrap();
        
        let mut field = create_test_field(&grid, angle);
        
        group.bench_with_input(
            BenchmarkId::new("Standard PML", angle),
            &angle,
            |b, _| {
                b.iter(|| {
                    let mut field_copy = field.clone();
                    for _ in 0..10 {
                        standard_pml.apply_acoustic(&mut field_copy, &grid, 0).unwrap();
                    }
                    black_box(field_copy);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("C-PML Grazing", angle),
            &angle,
            |b, _| {
                b.iter(|| {
                    let mut field_copy = field.clone();
                    for _ in 0..10 {
                        cpml_grazing.apply_acoustic(&mut field_copy, &grid, 0).unwrap();
                    }
                    black_box(field_copy);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Variable Update");
    
    let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);
    let config = CPMLConfig::default();
    let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
    
    let gradient = Array3::ones((128, 128, 128));
    
    group.bench_function("Update X memory", |b| {
        b.iter(|| {
            cpml.update_acoustic_memory(&gradient, 0).unwrap();
        });
    });
    
    group.bench_function("Update Y memory", |b| {
        b.iter(|| {
            cpml.update_acoustic_memory(&gradient, 1).unwrap();
        });
    });
    
    group.bench_function("Update Z memory", |b| {
        b.iter(|| {
            cpml.update_acoustic_memory(&gradient, 2).unwrap();
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_pml_comparison, bench_grazing_angles, bench_memory_update);
criterion_main!(benches);