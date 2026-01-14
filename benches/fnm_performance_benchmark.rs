//! Fast Nearfield Method Performance Benchmark
//!
//! This benchmark compares the performance of the Fast Nearfield Method (FNM)
//! against a simplified Rayleigh-Sommerfeld implementation to demonstrate
//! the O(n) vs O(n²) complexity advantage.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::{
    domain::source::RectangularTransducer,
    solver::analytical::transducer::{FNMConfig, FastNearfieldSolver},
};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Simplified Rayleigh-Sommerfeld implementation for comparison
///
/// # TODO: SIMPLIFIED BENCHMARK - NOT PRODUCTION CODE
/// This is a reference implementation for O(n²) complexity comparison only.
/// Real Rayleigh-Sommerfeld integral requires:
/// - Full spatial integration over transducer surface
/// - Green's function: G(r) = (1/4π) * exp(ikr)/r with obliquity factor
/// - Proper handling of near-field, far-field, and transition regions
/// - Element subdivision for accurate numerical integration
/// This simplified version is for benchmark timing purposes to demonstrate FNM speedup.
struct RayleighSommerfeldSolver {
    transducer: RectangularTransducer,
    c0: f64,
    rho0: f64,
}

impl RayleighSommerfeldSolver {
    fn new(transducer: RectangularTransducer, c0: f64, rho0: f64) -> Self {
        Self {
            transducer,
            c0,
            rho0,
        }
    }

    /// Compute pressure field using direct Rayleigh-Sommerfeld integration
    /// This is O(n²) complexity where n is the number of transducer elements
    fn compute_field(&self, velocity: &Array2<Complex64>, z: f64) -> Array2<Complex64> {
        let (n_elem_x, n_elem_y) = velocity.dim();
        let mut pressure = Array2::<Complex64>::zeros((n_elem_x, n_elem_y));

        let k = self.transducer.wavenumber(self.c0);
        let (elem_width, elem_height) = self.transducer.element_size();

        // For each observation point (on transducer surface for simplicity)
        for i in 0..n_elem_x {
            for j in 0..n_elem_y {
                let mut p_sum = Complex64::new(0.0, 0.0);

                // Sum contributions from all source elements (O(n²) loop)
                for si in 0..n_elem_x {
                    for sj in 0..n_elem_y {
                        // Distance from source to observation point
                        let dx = (i as f64 - si as f64) * elem_width;
                        let dy = (j as f64 - sj as f64) * elem_height;
                        let dz = z;

                        let r = (dx * dx + dy * dy + dz * dz).sqrt();

                        if r > 1e-12 {
                            // Avoid singularity
                            // Rayleigh-Sommerfeld Green's function
                            let green = Complex64::new(0.0, k * r).exp() / r;
                            let obliquity = dz / r; // z-component of unit vector

                            p_sum += velocity[[si, sj]] * green * obliquity;
                        }
                    }
                }

                // Scaling factor
                let scaling = Complex64::new(0.0, self.rho0 * self.c0 * k / (2.0 * PI));
                pressure[[i, j]] = p_sum * scaling;
            }
        }

        pressure
    }
}

fn benchmark_fnm_vs_rayleigh_sommerfeld(c: &mut Criterion) {
    let mut group = c.benchmark_group("FNM vs Rayleigh-Sommerfeld");

    // Test with different transducer sizes
    let transducer_sizes = vec![(8, 8), (12, 12), (16, 16)];

    for (nx, ny) in transducer_sizes {
        let transducer = RectangularTransducer {
            width: 5.0e-3,
            height: 5.0e-3,
            frequency: 2.0e6,
            elements: (nx, ny),
        };

        // Setup FNM solver
        let config = FNMConfig {
            angular_spectrum_size: (64, 64), // Smaller for benchmarking
            dx: 0.2e-3,                      // Coarser grid for speed
            ..Default::default()
        };

        let mut fnm_solver = FastNearfieldSolver::new(config).unwrap();
        fnm_solver.set_transducer(transducer.clone());
        fnm_solver.set_medium(1500.0, 1000.0);

        // Setup Rayleigh-Sommerfeld solver
        let rs_solver = RayleighSommerfeldSolver::new(transducer, 1500.0, 1000.0);

        // Create test velocity distribution
        let velocity = Array2::<Complex64>::from_elem((nx, ny), Complex64::new(1.0, 0.0));
        let z = 25e-3; // 25 mm

        // Precompute FNM factors
        fnm_solver.precompute_factors(z).unwrap();

        group.bench_function(format!("FNM_{}x{}", nx, ny), |b| {
            b.iter(|| {
                black_box(fnm_solver.compute_field(&velocity, z).unwrap());
            });
        });

        group.bench_function(format!("Rayleigh-Sommerfeld_{}x{}", nx, ny), |b| {
            b.iter(|| {
                black_box(rs_solver.compute_field(&velocity, z));
            });
        });
    }

    group.finish();
}

fn benchmark_fnm_precomputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("FNM Precomputation");

    let transducer = RectangularTransducer {
        width: 10.0e-3,
        height: 10.0e-3,
        frequency: 1.0e6,
        elements: (32, 32),
    };

    let config = FNMConfig {
        angular_spectrum_size: (128, 128),
        dx: 0.1e-3,
        ..Default::default()
    };

    let mut solver = FastNearfieldSolver::new(config).unwrap();
    solver.set_transducer(transducer);

    let z_distances = vec![10e-3, 25e-3, 50e-3, 100e-3];

    for &z in &z_distances {
        group.bench_function(format!("FNM_Precompute_{:.0}mm", z * 1000.0), |b| {
            b.iter(|| {
                // Clear cache to force recomputation
                solver.clear_cache();
                solver.precompute_factors(z).unwrap();
                black_box(());
            });
        });
    }

    group.finish();
}

fn benchmark_fnm_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("FNM Memory Scaling");

    let transducer_sizes = vec![(16, 16), (24, 24), (32, 32), (40, 40)];

    for (nx, ny) in transducer_sizes {
        let transducer = RectangularTransducer {
            width: 10.0e-3,
            height: 10.0e-3,
            frequency: 1.0e6,
            elements: (nx, ny),
        };

        let config = FNMConfig {
            angular_spectrum_size: (128, 128),
            dx: 0.1e-3,
            ..Default::default()
        };

        let mut solver = FastNearfieldSolver::new(config).unwrap();
        solver.set_transducer(transducer);

        // Precompute for multiple z-distances to test memory usage
        let z_values = vec![25e-3, 50e-3, 75e-3, 100e-3];

        group.bench_function(format!("FNM_Memory_{}x{}", nx, ny), |b| {
            b.iter(|| {
                solver.clear_cache();
                for &z in &z_values {
                    solver.precompute_factors(z).unwrap();
                    black_box(());
                }
                black_box(solver.memory_usage());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_fnm_vs_rayleigh_sommerfeld,
    benchmark_fnm_precomputation,
    benchmark_fnm_memory_scaling
);
criterion_main!(benches);
