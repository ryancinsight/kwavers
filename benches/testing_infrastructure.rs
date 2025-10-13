//! Comprehensive benchmarking suite for kwavers testing infrastructure
//!
//! **Purpose**: Measure and validate test execution performance
//! **Target**: Ensure all tests complete within SRS NFR-002 (<30s) requirements
//! **Methodology**: Criterion-based statistical benchmarking

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kwavers::{Grid, medium::HomogeneousMedium};
use kwavers::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
use kwavers::testing::acoustic_properties::*;
use kwavers::testing::medium_properties::*;
use kwavers::testing::grid_properties::*;

/// Benchmark grid creation for various sizes
fn bench_grid_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_creation");
    
    for size in [16, 32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter(|| {
                    let grid = Grid::new(
                        size, size, size,
                        0.001, 0.001, 0.001
                    ).expect("Grid creation");
                    black_box(grid);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark medium property validation
fn bench_medium_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_validation");
    
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001)
        .expect("Test grid");
    
    let medium = HomogeneousMedium::new(
        DENSITY_WATER,
        SOUND_SPEED_WATER,
        0.5,
        0.0,
        &grid
    );
    
    group.bench_function("verify_properties", |b| {
        b.iter(|| {
            let result = verify_medium_properties_physically_valid(&medium, &grid);
            black_box(result);
        });
    });
    
    group.finish();
}

/// Benchmark grid indexing operations
fn bench_grid_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_indexing");
    
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001)
        .expect("Test grid");
    
    group.bench_function("verify_safe_indexing", |b| {
        b.iter(|| {
            let result = verify_grid_indexing_safe(&grid);
            black_box(result);
        });
    });
    
    group.bench_function("coordinate_conversion", |b| {
        b.iter(|| {
            for i in (0..64).step_by(8) {
                for j in (0..64).step_by(8) {
                    for k in (0..64).step_by(8) {
                        let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                        let indices = grid.position_to_indices(x, y, z);
                        black_box(indices);
                    }
                }
            }
        });
    });
    
    group.finish();
}

/// Benchmark physical constraint validations
fn bench_physical_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("physical_constraints");
    
    group.bench_function("density_validation", |b| {
        b.iter(|| {
            for density in [1.0, 100.0, 1000.0, 5000.0].iter() {
                let valid = is_valid_density(*density);
                black_box(valid);
            }
        });
    });
    
    group.bench_function("sound_speed_validation", |b| {
        b.iter(|| {
            for speed in [100.0, 500.0, 1500.0, 6000.0].iter() {
                let valid = is_valid_sound_speed(*speed);
                black_box(valid);
            }
        });
    });
    
    group.bench_function("impedance_calculation", |b| {
        b.iter(|| {
            for &density in [1000.0, 1500.0, 2000.0].iter() {
                for &speed in [1000.0, 1500.0, 2000.0].iter() {
                    let valid = is_valid_acoustic_impedance(density, speed);
                    black_box(valid);
                }
            }
        });
    });
    
    group.finish();
}

/// Benchmark CFL condition calculations
fn bench_cfl_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfl_calculations");
    
    group.bench_function("cfl_stability_check", |b| {
        b.iter(|| {
            let c0 = SOUND_SPEED_WATER;
            let dx = 0.001;
            
            for dt_factor in [0.1, 0.5, 1.0, 2.0].iter() {
                let dt = dx / (c0 * 3.0f64.sqrt()) * dt_factor;
                let cfl = c0 * dt / dx;
                let stable = cfl < 1.0 / 3.0f64.sqrt();
                black_box(stable);
            }
        });
    });
    
    group.finish();
}

/// Benchmark absorption coefficient calculations
fn bench_absorption_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("absorption_calculations");
    
    group.bench_function("power_law_absorption", |b| {
        b.iter(|| {
            let alpha_0: f64 = 0.5;
            let f0: f64 = 1e6;
            
            for power in [0.5, 1.0, 1.5, 2.0].iter() {
                for freq in [0.5e6, 1e6, 2e6, 5e6].iter() {
                    let freq_ratio = freq / f0;
                    let alpha = alpha_0 * freq_ratio.powf(*power);
                    black_box(alpha);
                }
            }
        });
    });
    
    group.finish();
}

/// Benchmark wave propagation calculations
fn bench_wave_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("wave_propagation");
    
    group.bench_function("wavelength_calculation", |b| {
        b.iter(|| {
            let c0 = SOUND_SPEED_WATER;
            
            for freq in [0.5e6, 1e6, 2e6, 5e6, 10e6].iter() {
                let wavelength = c0 / freq;
                black_box(wavelength);
            }
        });
    });
    
    group.bench_function("wave_number_calculation", |b| {
        b.iter(|| {
            let c0 = SOUND_SPEED_WATER;
            
            for freq in [0.5e6, 1e6, 2e6, 5e6, 10e6].iter() {
                let omega = 2.0 * std::f64::consts::PI * freq;
                let k = omega / c0;
                black_box(k);
            }
        });
    });
    
    group.finish();
}

/// Benchmark energy conservation checks
fn bench_energy_conservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_conservation");
    
    group.bench_function("energy_density_calculation", |b| {
        b.iter(|| {
            let c0 = SOUND_SPEED_WATER;
            let rho0 = DENSITY_WATER;
            
            for pressure in [1e3, 1e4, 1e5, 1e6].iter() {
                let velocity = pressure / (rho0 * c0);
                let kinetic = 0.5 * rho0 * velocity.powi(2);
                let potential = pressure.powi(2) / (2.0 * rho0 * c0.powi(2));
                let total = kinetic + potential;
                black_box(total);
            }
        });
    });
    
    group.finish();
}

/// Benchmark acoustic impedance calculations
fn bench_impedance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("impedance_calculations");
    
    group.bench_function("impedance_reflection", |b| {
        b.iter(|| {
            // Water-muscle interface
            let z1 = DENSITY_WATER * SOUND_SPEED_WATER;
            let z2 = 1050.0 * 1547.0;
            
            let r = (z2 - z1) / (z2 + z1);
            let t = 1.0 + r;
            
            let r_intensity = r.powi(2);
            let t_intensity = (z1 / z2) * t.powi(2);
            
            black_box((r_intensity, t_intensity));
        });
    });
    
    group.finish();
}

/// Benchmark full validation suite execution time
fn bench_validation_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_suite");
    
    group.bench_function("complete_validation", |b| {
        b.iter(|| {
            // Simulate running all validation checks
            let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001)
                .expect("Grid");
            let medium = HomogeneousMedium::new(
                DENSITY_WATER,
                SOUND_SPEED_WATER,
                0.5,
                0.0,
                &grid
            );
            
            // Property validations
            let _ = verify_medium_properties_physically_valid(&medium, &grid);
            let _ = verify_grid_indexing_safe(&grid);
            let _ = is_valid_density(DENSITY_WATER);
            let _ = is_valid_sound_speed(SOUND_SPEED_WATER);
            let _ = is_valid_acoustic_impedance(DENSITY_WATER, SOUND_SPEED_WATER);
            
            black_box(());
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_grid_creation,
    bench_medium_validation,
    bench_grid_indexing,
    bench_physical_constraints,
    bench_cfl_calculations,
    bench_absorption_calculations,
    bench_wave_propagation,
    bench_energy_conservation,
    bench_impedance_calculations,
    bench_validation_suite
);

criterion_main!(benches);
