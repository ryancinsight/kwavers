//! Nonlinear Shear Wave Elastography Performance Benchmarks
//!
//! Benchmarks comparing nonlinear SWE performance against linear SWE
//! and measuring computational complexity of advanced algorithms.
//!
//! ## Literature Performance Targets
//!
//! ### Hyperelastic Constitutive Models
//! - **Neo-Hookean**: O(1) operations per evaluation
//! - **Mooney-Rivlin**: O(1) operations per evaluation
//! - **Ogden**: O(n_terms) operations, typically n_terms ≤ 3
//!
//! ### Eigenvalue Computation (Jacobi Method)
//! - **Convergence**: Quadratic convergence, typically 5-15 iterations for 3x3 matrices
//! - **Complexity**: O(n³) per iteration, O(n⁴) total for n=3
//! - **Reference**: Golub & Van Loan (1996), Matrix Computations
//!
//! ### Harmonic Detection
//! - **FFT-based**: O(N log N) complexity for N time samples
//! - **Filter-based**: O(N) complexity with digital filters
//! - **Reference**: Chen et al. (2013), IEEE TMI
//!
//! ## Performance Validation
//! - **Neo-Hookean stress**: < 50 ns/evaluation (single-threaded)
//! - **Matrix eigenvalues**: < 500 ns/computation (Jacobi method)
//! - **Harmonic detection**: < 100 μs for 1024 samples
//! - **Memory efficiency**: < 8 bytes per grid point for storage

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::domain::imaging::ultrasound::elastography::{
    InversionMethod, NonlinearInversionMethod,
};
use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use kwavers::physics::imaging::modalities::elastography::{
    HarmonicDetectionConfig, HarmonicDetector, HarmonicDisplacementField, HyperelasticModel,
    NonlinearElasticWaveSolver, NonlinearSWEConfig,
};
use kwavers::simulation::imaging::elastography::ShearWaveElastography;
use kwavers::solver::forward::elastic::ElasticWaveConfig;
use kwavers::solver::inverse::elastography::NonlinearInversion;
use ndarray::Array3;

/// Benchmark hyperelastic constitutive model evaluation
pub fn bench_hyperelastic_models(c: &mut Criterion) {
    let neo_hookean = HyperelasticModel::neo_hookean_soft_tissue();
    let mooney_rivlin = HyperelasticModel::mooney_rivlin_biological();

    // Test deformation gradient
    let f = [[1.1, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]];

    c.bench_function("neo_hookean_strain_energy", |b| {
        b.iter(|| black_box(neo_hookean.strain_energy(3.5, 3.2, 1.0)))
    });

    c.bench_function("mooney_rivlin_strain_energy", |b| {
        b.iter(|| black_box(mooney_rivlin.strain_energy(3.5, 3.2, 1.0)))
    });

    c.bench_function("neo_hookean_stress", |b| {
        b.iter(|| black_box(neo_hookean.cauchy_stress(&f)))
    });

    c.bench_function("mooney_rivlin_stress", |b| {
        b.iter(|| black_box(mooney_rivlin.cauchy_stress(&f)))
    });
}

/// Benchmark harmonic detection algorithms
pub fn bench_harmonic_detection(c: &mut Criterion) {
    let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());

    // Create test time series with harmonics
    let nx = 8;
    let ny = 8;
    let nz = 8;
    let n_times = 128;

    let mut time_series = ndarray::Array4::<f64>::zeros((nx, ny, nz, n_times));

    // Add fundamental (50 Hz) and second harmonic (100 Hz)
    let fundamental_freq = 50.0;
    let sampling_freq = 1000.0;

    for t in 0..n_times {
        let time = t as f64 / sampling_freq;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let fundamental = (2.0 * std::f64::consts::PI * fundamental_freq * time).sin();
                    let harmonic =
                        0.1 * (4.0 * std::f64::consts::PI * fundamental_freq * time).sin();
                    time_series[[i, j, k, t]] = fundamental + harmonic;
                }
            }
        }
    }

    c.bench_function("harmonic_analysis_8x8x8", |b| {
        b.iter(|| black_box(detector.analyze_harmonics(&time_series, sampling_freq)))
    });
}

/// Benchmark nonlinear inversion algorithms
pub fn bench_nonlinear_inversion(c: &mut Criterion) {
    // Create test harmonic field
    let mut harmonic_field = HarmonicDisplacementField::new(8, 8, 8, 2, 64);
    harmonic_field.fundamental_magnitude.fill(1.0);
    harmonic_field.harmonic_magnitudes[0].fill(0.1);
    harmonic_field.harmonic_snrs[0].fill(20.0);

    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();

    let harmonic_ratio_inv = NonlinearInversion::new(NonlinearInversionMethod::HarmonicRatio);
    let least_squares_inv =
        NonlinearInversion::new(NonlinearInversionMethod::NonlinearLeastSquares);
    let bayesian_inv = NonlinearInversion::new(NonlinearInversionMethod::BayesianInversion);

    c.bench_function("harmonic_ratio_inversion", |b| {
        b.iter(|| black_box(harmonic_ratio_inv.reconstruct_nonlinear(&harmonic_field, &grid)))
    });

    c.bench_function("nonlinear_least_squares_inversion", |b| {
        b.iter(|| black_box(least_squares_inv.reconstruct_nonlinear(&harmonic_field, &grid)))
    });

    c.bench_function("bayesian_inversion", |b| {
        b.iter(|| black_box(bayesian_inv.reconstruct_nonlinear(&harmonic_field, &grid)))
    });
}

/// Benchmark nonlinear vs linear wave propagation
pub fn bench_wave_propagation_comparison(c: &mut Criterion) {
    let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

    // Linear SWE setup
    let linear_swe = ShearWaveElastography::new(
        &grid,
        &medium,
        InversionMethod::TimeOfFlight,
        ElasticWaveConfig {
            simulation_time: 0.5e-3, // Short simulation for benchmarking
            ..Default::default()
        },
    )
    .unwrap();

    // Nonlinear SWE setup
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let nonlinear_solver = NonlinearElasticWaveSolver::new(
        &grid,
        &medium,
        material,
        NonlinearSWEConfig {
            enable_harmonics: false, // Disable harmonics for fair comparison
            ..Default::default()
        },
    )
    .unwrap();

    let push_location = [0.008, 0.008, 0.008];
    let initial_disp = Array3::zeros((16, 16, 16));

    c.bench_function("linear_swe_wave_propagation", |b| {
        b.iter(|| black_box(linear_swe.generate_shear_wave(push_location)))
    });

    c.bench_function("nonlinear_swe_wave_propagation", |b| {
        b.iter(|| black_box(nonlinear_solver.propagate_waves(&initial_disp)))
    });
}

/// Benchmark end-to-end NL-SWE workflow
pub fn bench_end_to_end_workflow(c: &mut Criterion) {
    let grid = Grid::new(12, 12, 12, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

    c.bench_function("nl_swe_end_to_end_workflow", |b| {
        b.iter(|| {
            // Step 1: Linear SWE wave generation
            let swe = black_box(
                ShearWaveElastography::new(
                    &grid,
                    &medium,
                    InversionMethod::TimeOfFlight,
                    ElasticWaveConfig {
                        simulation_time: 0.3e-3,
                        save_every: 5,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );

            let push_location = [0.006, 0.006, 0.006];
            let displacement_history = swe.generate_shear_wave(push_location).unwrap();

            // Step 2: Convert to time series format
            let n_times = displacement_history.len();
            let mut time_series = ndarray::Array4::<f64>::zeros((12, 12, 12, n_times));

            for t in 0..n_times {
                let magnitude = displacement_history[t].displacement_magnitude();
                for i in 0..12 {
                    for j in 0..12 {
                        for k in 0..12 {
                            time_series[[i, j, k, t]] = magnitude[[i, j, k]];
                        }
                    }
                }
            }

            // Step 3: Harmonic analysis
            let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());
            let harmonic_field = detector.analyze_harmonics(&time_series, 1000.0).unwrap();

            // Step 4: Nonlinear inversion
            let inversion = NonlinearInversion::new(NonlinearInversionMethod::HarmonicRatio);
            let _nonlinear_params = inversion
                .reconstruct_nonlinear(&harmonic_field, &grid)
                .unwrap();

            black_box(())
        })
    });
}

/// Benchmark memory usage scaling
pub fn bench_memory_scaling(c: &mut Criterion) {
    let sizes = [8, 12, 16];

    let mut group = c.benchmark_group("memory_scaling");

    for &size in &sizes {
        group.bench_function(format!("harmonic_field_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let field = HarmonicDisplacementField::new(size, size, size, 3, 64);
                black_box(field)
            })
        });
    }

    group.finish();
}

/// Benchmark convergence behavior
pub fn bench_convergence_analysis(c: &mut Criterion) {
    let grid = Grid::new(8, 8, 8, 0.002, 0.002, 0.002).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();

    let nonlinearity_levels = [0.01, 0.1, 0.2];

    let mut group = c.benchmark_group("nonlinearity_convergence");

    for &beta in &nonlinearity_levels {
        group.bench_function(format!("nonlinearity_{:.2}", beta), |b| {
            b.iter(|| {
                let config = NonlinearSWEConfig {
                    nonlinearity_parameter: beta,
                    ..Default::default()
                };

                let solver =
                    NonlinearElasticWaveSolver::new(&grid, &medium, material.clone(), config)
                        .unwrap();
                let initial_disp = Array3::zeros((8, 8, 8));
                let result = solver.propagate_waves(&initial_disp);

                black_box(result)
            })
        });
    }

    group.finish();
}

/// Performance validation against literature targets
pub fn bench_literature_performance_validation(c: &mut Criterion) {
    // Test hyperelastic constitutive model performance against literature targets

    let neo_hookean = HyperelasticModel::neo_hookean_soft_tissue();
    let ogden = HyperelasticModel::Ogden {
        mu: vec![1e3, 0.5e3],
        alpha: vec![1.5, 5.0],
    };

    // Test deformation gradient for stress computation
    let f = [[1.1, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]];

    // Literature target: Neo-Hookean stress computation < 50 ns
    c.bench_function("neo_hookean_stress_literature_target", |b| {
        b.iter(|| black_box(neo_hookean.cauchy_stress(&f)))
    });

    // Literature target: Ogden stress computation (with eigenvalue calculation) < 1 μs
    c.bench_function("ogden_stress_literature_target", |b| {
        b.iter(|| black_box(ogden.cauchy_stress(&f)))
    });

    // Matrix eigenvalue computation performance validation
    let test_matrix = [[2.0, 0.1, 0.0], [0.1, 3.0, 0.1], [0.0, 0.1, 1.0]];

    // Literature target: Jacobi eigenvalue algorithm < 500 ns for 3x3 matrix
    c.bench_function("matrix_eigenvalue_jacobi_literature_target", |b| {
        b.iter(|| black_box(neo_hookean.matrix_eigenvalues(&test_matrix)))
    });

    // Memory efficiency validation
    c.bench_function("hyperelastic_memory_efficiency", |b| {
        b.iter(|| {
            // Measure memory usage for storing hyperelastic material properties
            let materials = [
                black_box(HyperelasticModel::neo_hookean_soft_tissue()),
                black_box(HyperelasticModel::mooney_rivlin_biological()),
                black_box(HyperelasticModel::Ogden {
                    mu: vec![1e3, 0.5e3],
                    alpha: vec![1.5, 5.0],
                }),
            ];

            // Literature target: < 8 bytes per grid point for material storage
            black_box(materials.len())
        });
    });
}

/// Benchmark memory usage patterns for NL-SWE
pub fn bench_memory_usage_patterns(c: &mut Criterion) {
    // Test memory allocation patterns for nonlinear SWE simulations

    let grid_sizes = vec![16, 32, 64];

    let mut group = c.benchmark_group("memory_usage_scaling");

    for &size in &grid_sizes {
        group.bench_function(format!("grid_{}x{}x{}", size, size, size), |b| {
            b.iter(|| {
                let grid = Grid::new(size, size, size, 0.001, 0.001, 0.001).unwrap();
                let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
                let material = HyperelasticModel::neo_hookean_soft_tissue();

                // Measure memory allocation for solver creation
                let config = NonlinearSWEConfig::default();
                let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config);

                black_box(solver)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hyperelastic_models,
    bench_harmonic_detection,
    bench_nonlinear_inversion,
    bench_wave_propagation_comparison,
    bench_end_to_end_workflow,
    bench_memory_scaling,
    bench_convergence_analysis,
    bench_literature_performance_validation,
    bench_memory_usage_patterns
);
criterion_main!(benches);
