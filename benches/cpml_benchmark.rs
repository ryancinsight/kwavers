//! Benchmark comparing C-PML and standard PML performance
//!
//! Comprehensive comparison of Convolutional PML vs standard PML boundary conditions
//! across different grid sizes, frequencies, and absorptive properties.
//!
//! References:
//! - Roden & Gedney (2000) "Convolutional PML (CPML): An efficient FDTD implementation"
//! - Berenger (1994) "A perfectly matched layer for the absorption of electromagnetic waves"
//! - Komatitsch & Martin (2007) "An unsplit convolutional PML"

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers::domain::boundary::cpml::{CPMLBoundary, CPMLConfig};
use kwavers::domain::boundary::pml::{PMLBoundary, PMLConfig};
use kwavers::domain::boundary::Boundary;
use kwavers::grid::Grid;
use ndarray::Array3;

/// Benchmark CPML gradient correction for various grid sizes
fn cpml_gradient_correction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpml_gradient_correction");

    for size in [32, 64, 128].iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");
        let config = CPMLConfig::with_thickness(10);
        let mut boundary = CPMLBoundary::new(config, &grid, 1500.0)
            .expect("CPML boundary creation should succeed");
        let mut gradient = Array3::zeros((grid.nx, grid.ny, grid.nz));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                boundary.update_and_apply_gradient_correction(&mut gradient, 0);
                black_box(&gradient);
            });
        });
    }

    group.finish();
}

/// Benchmark CPML 4D field update for various grid sizes
fn cpml_field_update_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpml_field_update");

    for size in [32, 64, 128].iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");
        let config = CPMLConfig::with_thickness(10);
        let mut boundary = CPMLBoundary::new(config, &grid, 1500.0)
            .expect("CPML boundary creation should succeed");
        let mut field = grid.create_field();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                boundary
                    .apply_acoustic(field.view_mut(), &grid, 0)
                    .expect("CPML field update should succeed");
                black_box(&field);
            });
        });
    }

    group.finish();
}

/// Benchmark standard PML boundary application for various grid sizes
fn pml_apply_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pml_apply");

    for size in [32, 64, 128].iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");
        let config = PMLConfig::default();
        let mut boundary = PMLBoundary::new(config).expect("PML boundary creation should succeed");
        let mut field = grid.create_field();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                boundary
                    .apply_acoustic(field.view_mut(), &grid, 0)
                    .expect("PML apply should succeed");
                black_box(&field);
            });
        });
    }

    group.finish();
}

/// Benchmark PML frequency domain application
fn pml_freq_domain_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pml_freq_domain");

    for size in [32, 64, 128].iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");
        let config = PMLConfig::default();
        let mut boundary = PMLBoundary::new(config).expect("PML boundary creation should succeed");
        let mut field_freq = Array3::from_elem(
            (grid.nx, grid.ny, grid.nz),
            rustfft::num_complex::Complex::new(0.0, 0.0),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                boundary
                    .apply_acoustic_freq(&mut field_freq, &grid, 0)
                    .expect("PML freq domain apply should succeed");
                black_box(&field_freq);
            });
        });
    }

    group.finish();
}

/// Compare CPML vs PML with varying thickness
fn cpml_vs_pml_thickness_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpml_vs_pml_thickness");

    let sizes = [32, 64, 128];
    for size in sizes.iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");

        // CPML with varying thickness
        for thickness in [5, 10, 20].iter() {
            let config = CPMLConfig::with_thickness(*thickness);
            let mut boundary = CPMLBoundary::new(config, &grid, 1500.0)
                .expect("CPML boundary creation should succeed");
            let mut gradient = Array3::zeros((grid.nx, grid.ny, grid.nz));

            group.bench_with_input(
                BenchmarkId::new("cpml", format!("{}_{}", size, thickness)),
                &(*size, *thickness),
                |b, _| {
                    b.iter(|| {
                        boundary.update_and_apply_gradient_correction(&mut gradient, 0);
                        black_box(&gradient);
                    });
                },
            );
        }

        // Standard PML for comparison (uses internal thickness from config)
        let config = PMLConfig::default();
        let mut boundary = PMLBoundary::new(config).expect("PML boundary creation should succeed");
        let mut field = grid.create_field();

        group.bench_with_input(
            BenchmarkId::new("pml", format!("{}", size)),
            size,
            |b, _| {
                b.iter(|| {
                    boundary
                        .apply_acoustic(field.view_mut(), &grid, 0)
                        .expect("PML apply should succeed");
                    black_box(&field);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency of CPML vs PML creation
fn memory_usage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for size in [32, 64, 128].iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");

        // CPML memory allocation
        group.bench_with_input(BenchmarkId::new("cpml_creation", size), size, |b, _| {
            b.iter(|| {
                let config = CPMLConfig::with_thickness(10);
                let boundary = CPMLBoundary::new(config, &grid, 1500.0)
                    .expect("CPML boundary creation should succeed");
                black_box(boundary);
            });
        });

        // PML memory allocation
        group.bench_with_input(BenchmarkId::new("pml_creation", size), size, |b, _| {
            b.iter(|| {
                let config = PMLConfig::default();
                let boundary =
                    PMLBoundary::new(config).expect("PML boundary creation should succeed");
                black_box(boundary);
            });
        });
    }

    group.finish();
}

/// Benchmark CPML reset operation
fn cpml_reset_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpml_reset");

    for size in [32, 64, 128].iter() {
        let grid =
            Grid::new(*size, *size, *size, 1e-3, 1e-3, 1e-3).expect("Grid creation should succeed");
        let config = CPMLConfig::with_thickness(10);
        let mut boundary = CPMLBoundary::new(config, &grid, 1500.0)
            .expect("CPML boundary creation should succeed");

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                boundary.reset();
                black_box(&boundary);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    cpml_gradient_correction_benchmark,
    cpml_field_update_benchmark,
    pml_apply_benchmark,
    pml_freq_domain_benchmark,
    cpml_vs_pml_thickness_comparison,
    memory_usage_comparison,
    cpml_reset_benchmark
);

criterion_main!(benches);
