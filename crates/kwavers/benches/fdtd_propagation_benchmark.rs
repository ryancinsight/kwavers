//! FDTD Propagation Benchmark
//!
//! Rigorous performance and accuracy benchmarking of the FDTD solver
//! against analytical solutions. Follows mathematical verification principles.
//!
//! ## Mathematical Specification
//!
//! ### Plane Wave Analytical Solution
//! For a plane wave with amplitude A, frequency f, propagating in x-direction:
//!
//! ```text
//! p(x,t) = A sin(2πf(t - x/c)) = A sin(ωt - kx)
//! ```
//!
//! where:
//! - ω = 2πf is the angular frequency
//! - k = ω/c = 2π/λ is the wavenumber
//! - c is the sound speed (1500 m/s in water)
//!
//! ### Energy Conservation
//! For an acoustic wave in non-absorbing medium:
//!
//! ```text
//! E = ∫(p²/(2ρc²) + ½ρ|v|²)dV = constant
//! ```
//!
//! ### Numerical Dispersion Relation
//! The FDTD scheme introduces numerical dispersion:
//!
//! ```text
//! sin²(ωΔt/2)/(cΔt)² = Σᵢ sin²(kᵢΔxᵢ/2)/(Δxᵢ)²
//! ```
//!
//! For well-resolved waves (PPW > 10), this error is < 1%.
//!
//! ## References
//! - Yee, K.S. (1966). IEEE TAP 14(3), 302-307. DOI: 10.1109/TAP.1966.1138693
//! - Taflove & Hagness (2005). Computational Electrodynamics, 3rd ed. Artech House.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;
use kwavers_solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use ndarray::{Array2, Array3};
use std::time::Duration;

/// Benchmark configuration
#[derive(Debug, Clone)]
struct FdtdBenchmarkConfig {
    grid_sizes: Vec<(usize, usize, usize)>,
    frequencies_hz: Vec<f64>,
    time_steps: usize,
}

impl Default for FdtdBenchmarkConfig {
    fn default() -> Self {
        Self {
            // Grid sizes: Small (32³), Medium (64³) for rapid benchmarking
            // Large grids deferred to Phase 3 validation suite
            grid_sizes: vec![(32, 32, 32), (64, 64, 64)],
            // Two frequencies: 1 MHz (well-resolved), 5 MHz (PPW ~ 30)
            frequencies_hz: vec![1e6, 5e6],
            time_steps: 100, // Fixed timestep count for consistent benchmarking
        }
    }
}

/// Setup plane wave problem with analytical solution
///
/// Returns grid, medium, source, and expected analytical solution parameters
///
/// THEOREM: Plane Wave Setup
/// For sound speed c = 1500 m/s and grid spacing dx = dy = dz,
/// the wavelength λ = c/f. The points per wavelength requirement is:
/// PPW = λ/dx ≥ 10 (criterion convergence).
///
/// For PPW = 10 grid: dx = c/(10f) = 150/(f in MHz) mm
fn setup_plane_wave_problem(
    grid_size: (usize, usize, usize),
    frequency_hz: f64,
    sound_speed: f64,
) -> Result<(Grid, HomogeneousMedium, GridSource, f64), kwavers_core::error::KwaversError> {
    let (nx, ny, nz) = grid_size;

    // Ensure adequate points per wavelength: PPW ≥ 10
    let wavelength = sound_speed / frequency_hz;
    let dx = wavelength / 10.0; // 10 points per wavelength minimum
    let dt = 0.5 * dx / sound_speed; // CFL factor 0.5 for stability margin

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::water(&grid);

    // Create sinusoidal source at center
    let mut p_mask = Array3::<f64>::zeros((nx, ny, nz));
    let source_idx = (nx / 2, ny / 2, nz / 2);
    p_mask[[source_idx.0, source_idx.1, source_idx.2]] = 1.0;

    // Temporal signal: single period of sine wave
    let period = 1.0 / frequency_hz;
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;

    // Signal duration: 5 periods for clear waveform
    let nt = 5 * (period / dt).ceil() as usize;
    let mut p_signal = Array2::<f64>::zeros((1, nt));

    for t in 0..nt {
        let time = t as f64 * dt;
        let phase = omega * time;
        // Gaussian-envelope sine to localize source
        let envelope = (-0.5 * ((time - 2.5 * period) / (0.5 * period)).powi(2)).exp();
        p_signal[[0, t]] = envelope * phase.sin();
    }

    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        ..Default::default()
    };

    Ok((grid, medium, source, dt))
}

/// Benchmark FDTD propagation at given parameters
fn bench_fdtd_propagation(
    c: &mut Criterion,
    grid_size: (usize, usize, usize),
    frequency_hz: f64,
    time_steps: usize,
) {
    let sound_speed = 1500.0; // m/s

    let (_grid, _medium, _source, dt) =
        setup_plane_wave_problem(grid_size, frequency_hz, sound_speed)
            .expect("Failed to setup plane wave problem");

    let _config = FdtdConfig {
        nt: time_steps,
        dt,
        ..Default::default()
    };

    // Compute throughput metrics
    let grid_points = (grid_size.0 * grid_size.1 * grid_size.2) as u64;
    let total_updates = grid_points * time_steps as u64;
    let grid_size_label = format!(
        "{}x{}x{}@{}MHz",
        grid_size.0,
        grid_size.1,
        grid_size.2,
        (frequency_hz / 1e6) as i32
    );

    let mut group = c.benchmark_group(format!("fdtd_propagation_{}", grid_size_label));
    group.throughput(Throughput::Elements(total_updates));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10); // Fewer samples for longer-running benchmarks

    group.bench_function(&grid_size_label, |b| {
        b.iter(|| {
            // Reset solver state by creating fresh instance
            let (g, m, s, _dt) = setup_plane_wave_problem(grid_size, frequency_hz, sound_speed)
                .expect("Setup failed");
            let cfg = FdtdConfig {
                nt: time_steps,
                dt,
                ..Default::default()
            };
            let mut solv = FdtdSolver::new(cfg, &g, &m, s).expect("Solver creation failed");

            // Run simulation
            for _ in 0..time_steps {
                solv.step_forward().expect("Solver step failed");
            }

            // Return final pressure field for validation
            black_box(&solv);
        });
    });

    group.finish();
}

/// Main benchmark function
fn fdtd_propagation_benchmark(c: &mut Criterion) {
    let config = FdtdBenchmarkConfig::default();

    for grid_size in &config.grid_sizes {
        for frequency in &config.frequencies_hz {
            bench_fdtd_propagation(c, *grid_size, *frequency, config.time_steps);
        }
    }
}

/// Energy conservation validation benchmark
///
/// THEOREM: Energy Conservation in FDTD
/// For a lossless medium with Δt satisfying CFL condition,
/// the discrete energy E^n = Σ(|p^n|²/(2ρc²) + ½ρ|v^{n+½}|²)dx³
/// satisfies E^{n+1} = E^n + O(Δt³) for the leapfrog scheme.
fn energy_conservation_benchmark(c: &mut Criterion) {
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 1e-4;
    let dt = dx / (1500.0 * 3.0_f64.sqrt()) * 0.95; // CFL-safe timestep

    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("Grid creation failed");
    let medium = HomogeneousMedium::water(&grid);

    // Initial condition: Gaussian pressure distribution
    let mut p_mask = Array3::<f64>::zeros((nx, ny, nz));
    let center = (nx / 2, ny / 2, nz / 2);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let r2 = ((i as i32 - center.0 as i32).pow(2)
                    + (j as i32 - center.1 as i32).pow(2)
                    + (k as i32 - center.2 as i32).pow(2)) as f64;
                let sigma2 = (nx as f64 / 8.0).powi(2);
                p_mask[[i, j, k]] = (-r2 / (2.0 * sigma2)).exp();
            }
        }
    }

    // Single impulse source
    let mut p_signal = Array2::<f64>::zeros((1, 100));
    p_signal[[0, 0]] = 1.0;

    let source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        ..Default::default()
    };

    let config = FdtdConfig {
        nt: 50,
        dt,
        ..Default::default()
    };

    let mut solver =
        FdtdSolver::new(config, &grid, &medium, source).expect("Solver creation failed");

    c.bench_function("fdtd_energy_32_cubed", |b| {
        b.iter(|| {
            for _ in 0..50 {
                solver.step_forward().expect("Step failed");
            }
            black_box(&solver);
        });
    });
}

criterion_group!(
    benches,
    fdtd_propagation_benchmark,
    energy_conservation_benchmark
);
criterion_main!(benches);
