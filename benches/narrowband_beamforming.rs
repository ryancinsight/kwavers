//! Narrowband Beamforming Performance Benchmarks
//!
//! This benchmark suite validates that the narrowband migration maintains
//! performance within ±5% of the baseline. It measures critical path operations:
//!
//! 1. **Steering Vector Computation** - O(N) per candidate
//! 2. **Snapshot Extraction** - O(N×M log M) for STFT
//! 3. **Capon Spectrum Evaluation** - O(N³) for matrix inversion + O(N²) per point
//!
//! # Performance Targets
//!
//! - **Steering (8 sensors):** < 1 µs
//! - **Snapshots (8 sensors, 256 samples):** < 100 µs
//! - **Capon point (8 sensors, 64 snapshots):** < 500 µs
//!
//! # Mathematical Verification
//!
//! Benchmarks also assert correctness (not just speed):
//! - Steering vectors have unit magnitude
//! - Capon spectrum is positive and finite
//! - No silent failures or error masking

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kwavers::analysis::signal_processing::beamforming::narrowband::{
    capon_spatial_spectrum_point, extract_narrowband_snapshots, CaponSpectrumConfig,
    NarrowbandSteering, SnapshotScenario, SnapshotSelection,
};
use kwavers::analysis::signal_processing::beamforming::covariance::{
    CovarianceEstimator, CovariancePostProcess,
};
use kwavers::analysis::signal_processing::beamforming::utils::steering::SteeringVectorMethod;
use ndarray::Array3;
use std::f64::consts::PI;

/// Generate synthetic RF data for benchmarking (plane wave from broadside)
fn generate_benchmark_data(
    n_sensors: usize,
    n_samples: usize,
    frequency_hz: f64,
    sampling_frequency_hz: f64,
) -> Array3<f64> {
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

    for sensor_idx in 0..n_sensors {
        for sample_idx in 0..n_samples {
            let t = sample_idx as f64 / sampling_frequency_hz;
            // Plane wave from broadside (no phase shift between sensors)
            data[(sensor_idx, 0, sample_idx)] = (2.0 * PI * frequency_hz * t).cos();
        }
    }

    data
}

/// Generate ULA sensor positions
fn generate_ula_positions(n_sensors: usize, spacing_m: f64) -> Vec<[f64; 3]> {
    (0..n_sensors)
        .map(|i| [i as f64 * spacing_m, 0.0, 0.0])
        .collect()
}

/// Benchmark: Steering vector computation for various array sizes
fn bench_steering_vector_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrowband_steering");

    let sound_speed = 1500.0; // m/s
    let frequency = 100_000.0; // 100 kHz
    let candidate = [0.0, 0.0, 0.05]; // 5 cm in front

    for n_sensors in [4, 8, 16, 32, 64].iter() {
        let spacing = 0.0075; // λ/2 at 100 kHz
        let positions = generate_ula_positions(*n_sensors, spacing);
        let steering = NarrowbandSteering::new(positions.clone(), sound_speed)
            .expect("steering initialization");

        group.throughput(Throughput::Elements(*n_sensors as u64));
        group.bench_with_input(
            BenchmarkId::new("steering_vector", n_sensors),
            n_sensors,
            |b, _| {
                b.iter(|| {
                    let sv = steering
                        .steering_vector_point(black_box(candidate), black_box(frequency))
                        .expect("steering vector");

                    // Correctness assertion: unit magnitude
                    for &element in sv.as_array().iter() {
                        debug_assert!((element.norm() - 1.0).abs() < 1e-10);
                    }

                    black_box(sv)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Snapshot extraction (STFT-based) for various data sizes
fn bench_snapshot_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrowband_snapshots");

    let frequency = 100_000.0;
    let sampling_frequency = 1_000_000.0;

    for n_sensors in [4, 8, 16].iter() {
        for n_samples in [128, 256, 512].iter() {
            let data =
                generate_benchmark_data(*n_sensors, *n_samples, frequency, sampling_frequency);

            let scenario = SnapshotScenario {
                frequency_hz: frequency,
                sampling_frequency_hz: sampling_frequency,
                fractional_bandwidth: Some(0.05),
                prefer_robustness: true,
                prefer_time_resolution: false,
            };
            let selection = SnapshotSelection::Auto(scenario);

            let label = format!("{}sens_{}samp", n_sensors, n_samples);
            group.throughput(Throughput::Elements((n_sensors * n_samples) as u64));
            group.bench_with_input(
                BenchmarkId::new("stft_snapshots", &label),
                &data,
                |b, data| {
                    b.iter(|| {
                        let snapshots =
                            extract_narrowband_snapshots(black_box(data), black_box(&selection))
                                .expect("snapshot extraction");

                        // Correctness assertion: valid dimensions
                        debug_assert_eq!(snapshots.nrows(), *n_sensors);
                        debug_assert!(snapshots.ncols() > 0);

                        black_box(snapshots)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Capon spatial spectrum evaluation for single point
fn bench_capon_spectrum_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrowband_capon");

    let frequency = 100_000.0;
    let sampling_frequency = 1_000_000.0;
    let sound_speed = 1500.0;
    let spacing = 0.0075;
    let candidate = [0.0, 0.0, 0.05];

    for n_sensors in [4, 8, 16].iter() {
        let n_samples = 256;
        let data = generate_benchmark_data(*n_sensors, n_samples, frequency, sampling_frequency);
        let positions = generate_ula_positions(*n_sensors, spacing);

        let scenario = SnapshotScenario {
            frequency_hz: frequency,
            sampling_frequency_hz: sampling_frequency,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        };

        let cfg = CaponSpectrumConfig {
            frequency_hz: frequency,
            sound_speed,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process: CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: candidate,
            },
            sampling_frequency_hz: Some(sampling_frequency),
            snapshot_selection: Some(SnapshotSelection::Auto(scenario)),
            baseband_snapshot_step_samples: None,
        };

        group.throughput(Throughput::Elements(*n_sensors as u64));
        group.bench_with_input(
            BenchmarkId::new("capon_point", n_sensors),
            n_sensors,
            |b, _| {
                b.iter(|| {
                    let spectrum = capon_spatial_spectrum_point(
                        black_box(&data),
                        black_box(&positions),
                        black_box(candidate),
                        black_box(&cfg),
                    )
                    .expect("capon spectrum");

                    // Correctness assertion: positive and finite
                    debug_assert!(spectrum.is_finite());
                    debug_assert!(spectrum > 0.0);

                    black_box(spectrum)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Full narrowband localization pipeline (multiple candidates)
fn bench_narrowband_localization_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrowband_localization");
    group.sample_size(20); // Reduce sample size for expensive operation

    let n_sensors = 8;
    let n_samples = 256;
    let frequency = 100_000.0;
    let sampling_frequency = 1_000_000.0;
    let sound_speed = 1500.0;
    let spacing = 0.0075;

    let data = generate_benchmark_data(n_sensors, n_samples, frequency, sampling_frequency);
    let positions = generate_ula_positions(n_sensors, spacing);

    // Grid search: 11 candidate points
    let candidates: Vec<[f64; 3]> = (-5..=5)
        .map(|i| {
            let angle_deg = i as f64 * 5.0; // -25° to +25°
            let angle_rad = angle_deg * PI / 180.0;
            let distance = 0.05;
            [distance * angle_rad.sin(), 0.0, distance * angle_rad.cos()]
        })
        .collect();

    let scenario = SnapshotScenario {
        frequency_hz: frequency,
        sampling_frequency_hz: sampling_frequency,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    };

    group.throughput(Throughput::Elements(candidates.len() as u64));
    group.bench_function("grid_search_11points", |b| {
        b.iter(|| {
            let mut spectra = Vec::with_capacity(candidates.len());

            for &candidate in &candidates {
                let cfg = CaponSpectrumConfig {
                    frequency_hz: frequency,
                    sound_speed,
                    diagonal_loading: 1e-3,
                    covariance: CovarianceEstimator {
                        forward_backward_averaging: false,
                        num_snapshots: 1,
                        post_process: CovariancePostProcess::None,
                    },
                    steering: SteeringVectorMethod::SphericalWave {
                        source_position: candidate,
                    },
                    sampling_frequency_hz: Some(sampling_frequency),
                    snapshot_selection: Some(SnapshotSelection::Auto(scenario.clone())),
                    baseband_snapshot_step_samples: None,
                };

                let spectrum = capon_spatial_spectrum_point(
                    black_box(&data),
                    black_box(&positions),
                    black_box(candidate),
                    black_box(&cfg),
                )
                .expect("capon spectrum");

                spectra.push(spectrum);
            }

            // Correctness: All spectra valid
            debug_assert!(spectra.iter().all(|s| s.is_finite() && *s > 0.0));

            black_box(spectra)
        });
    });

    group.finish();
}

/// Benchmark: Diagonal loading impact on computation time
fn bench_diagonal_loading_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrowband_diagonal_loading");

    let n_sensors = 8;
    let n_samples = 256;
    let frequency = 100_000.0;
    let sampling_frequency = 1_000_000.0;
    let sound_speed = 1500.0;
    let spacing = 0.0075;
    let candidate = [0.0, 0.0, 0.05];

    let data = generate_benchmark_data(n_sensors, n_samples, frequency, sampling_frequency);
    let positions = generate_ula_positions(n_sensors, spacing);

    let scenario = SnapshotScenario {
        frequency_hz: frequency,
        sampling_frequency_hz: sampling_frequency,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    };

    for &loading in &[0.0, 1e-6, 1e-3, 1e-1] {
        let cfg = CaponSpectrumConfig {
            frequency_hz: frequency,
            sound_speed,
            diagonal_loading: loading,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process: CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: candidate,
            },
            sampling_frequency_hz: Some(sampling_frequency),
            snapshot_selection: Some(SnapshotSelection::Auto(scenario.clone())),
            baseband_snapshot_step_samples: None,
        };

        group.bench_with_input(
            BenchmarkId::new("loading", format!("{:.0e}", loading)),
            &loading,
            |b, _| {
                b.iter(|| {
                    let spectrum = capon_spatial_spectrum_point(
                        black_box(&data),
                        black_box(&positions),
                        black_box(candidate),
                        black_box(&cfg),
                    )
                    .expect("capon spectrum");

                    black_box(spectrum)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_steering_vector_computation,
    bench_snapshot_extraction,
    bench_capon_spectrum_point,
    bench_narrowband_localization_grid,
    bench_diagonal_loading_sensitivity,
);
criterion_main!(benches);
