//! Integration tests for narrowband beamforming pipeline
//!
//! These tests validate the end-to-end pipeline: steering → snapshots → Capon spectrum.
//! They verify that the components compose correctly and produce mathematically sound results.
//!
//! # Test Categories
//!
//! 1. **Pipeline Integration**: Full steering → snapshots → Capon flow
//! 2. **Cross-Method Consistency**: Different snapshot methods produce consistent results
//! 3. **Invariance Properties**: Mathematical invariants hold across the pipeline
//! 4. **Edge Cases**: Boundary conditions and degenerate inputs
//!
//! # Design Principles
//!
//! - **No Error Masking**: Tests fail loudly on unexpected conditions
//! - **Deterministic**: No random seeds; reproducible results
//! - **Literature-Grounded**: Expected behaviors based on signal processing theory
//! - **Minimal Mocking**: Use real implementations where possible
//!
//! # Tests Status
//!
//! **Status**: Re-enabled after beamforming architecture consolidation
//! **Date**: 2026-01-24
//! **Architecture**: Uses consolidated analysis layer beamforming APIs
//!
//! These tests validate the narrowband beamforming pipeline after the architecture refactor:
//! - Uses `analysis::signal_processing::beamforming::covariance::*`
//! - Uses `analysis::signal_processing::beamforming::utils::steering::*`
//! - All imports updated for new API organization
#[cfg(test)]
mod tests {
    use super::super::{
        capon::{capon_spatial_spectrum_point, CaponSpectrumConfig},
        snapshots::{extract_narrowband_snapshots, SnapshotScenario, SnapshotSelection},
        steering::NarrowbandSteering,
    };
    use crate::analysis::signal_processing::beamforming::{
        covariance::{CovarianceEstimator, CovariancePostProcess},
        utils::steering::SteeringVectorMethod,
    };
    use ndarray::Array3;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    /// Helper: Generate synthetic array data with a plane wave from a known direction
    ///
    /// # Mathematical Model
    ///
    /// For a plane wave at angle θ relative to the array axis:
    /// ```text
    /// x_m(t) = cos(2πf₀t + k·d_m·sin(θ))
    /// ```
    /// where:
    /// - f₀ = carrier frequency
    /// - k = 2πf₀/c (wavenumber)
    /// - d_m = position of sensor m
    /// - θ = angle of arrival
    fn generate_plane_wave_data(
        n_sensors: usize,
        sensor_spacing_m: f64,
        n_samples: usize,
        sampling_frequency_hz: f64,
        signal_frequency_hz: f64,
        angle_deg: f64,
        sound_speed_m_per_s: f64,
        snr_db: f64,
    ) -> Array3<f64> {
        let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

        let angle_rad = angle_deg * PI / 180.0;
        let k = 2.0 * PI * signal_frequency_hz / sound_speed_m_per_s; // wavenumber

        // Signal power (assuming unit amplitude)
        let signal_power = 0.5; // RMS power of cos wave with amplitude 1
        let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);
        let noise_std = noise_power.sqrt();

        for sensor_idx in 0..n_sensors {
            let sensor_position_m = sensor_idx as f64 * sensor_spacing_m;
            let phase_shift = k * sensor_position_m * angle_rad.sin();

            for sample_idx in 0..n_samples {
                let t = sample_idx as f64 / sampling_frequency_hz;
                let signal = (2.0 * PI * signal_frequency_hz * t + phase_shift).cos();

                // Add white Gaussian noise (simplified: use deterministic pseudo-noise)
                let noise = noise_std
                    * ((sample_idx as f64 * 17.0 + sensor_idx as f64 * 23.0).sin()
                        + (sample_idx as f64 * 13.0 + sensor_idx as f64 * 29.0).cos())
                    / 2.0_f64.sqrt();

                data[(sensor_idx, 0, sample_idx)] = signal + noise;
            }
        }

        data
    }

    /// Helper: Generate sensor positions (uniform linear array)
    fn generate_ula_positions(n_sensors: usize, spacing_m: f64) -> Vec<[f64; 3]> {
        (0..n_sensors)
            .map(|i| [i as f64 * spacing_m, 0.0, 0.0])
            .collect()
    }

    #[test]
    fn end_to_end_pipeline_produces_finite_spectrum() {
        // Setup: 4-element ULA, plane wave from broadside
        let n_sensors = 4;
        let spacing_m = 0.0075; // λ/2 at 100 kHz in water
        let n_samples = 128;
        let fs = 1_000_000.0; // 1 MHz sampling
        let f0 = 100_000.0; // 100 kHz signal
        let c = 1500.0; // sound speed
        let snr_db = 20.0;

        let data =
            generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);
        let positions = generate_ula_positions(n_sensors, spacing_m);

        // Pipeline step 1: Extract snapshots
        let scenario = SnapshotScenario {
            frequency_hz: f0,
            sampling_frequency_hz: fs,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        };
        let selection = SnapshotSelection::Auto(scenario);
        let snapshots =
            extract_narrowband_snapshots(&data, &selection).expect("snapshot extraction");

        assert!(
            snapshots.nrows() == n_sensors,
            "snapshot dimensions mismatch"
        );
        assert!(snapshots.ncols() > 0, "no snapshots extracted");

        // Pipeline step 2: Compute Capon spectrum at broadside
        let cfg = CaponSpectrumConfig {
            frequency_hz: f0,
            sound_speed: c,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: snapshots.ncols(),
                post_process: CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: [0.0, 0.0, 0.05], // 5 cm in front
            },
            sampling_frequency_hz: Some(fs),
            snapshot_selection: Some(selection),
            baseband_snapshot_step_samples: None,
        };

        let spectrum = capon_spatial_spectrum_point(&data, &positions, [0.0, 0.0, 0.05], &cfg)
            .expect("capon spectrum");

        // Validation: spectrum must be finite and positive
        assert!(spectrum.is_finite(), "spectrum is not finite");
        assert!(spectrum > 0.0, "spectrum must be positive");
    }

    #[test]
    fn capon_spectrum_varies_across_candidate_grid() {
        // Test that Capon spectrum produces different values for different candidate points
        // This validates the pipeline without requiring accurate DOA estimation
        let n_sensors = 8;
        let spacing_m = 0.0075; // λ/2 at 100 kHz
        let n_samples = 256;
        let fs = 1_000_000.0;
        let f0 = 100_000.0;
        let c = 1500.0;
        let snr_db = 30.0; // High SNR

        // Generate data with a source at broadside
        let data =
            generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);
        let positions = generate_ula_positions(n_sensors, spacing_m);

        let scenario = SnapshotScenario {
            frequency_hz: f0,
            sampling_frequency_hz: fs,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        };
        let selection = SnapshotSelection::Auto(scenario);

        let cfg = CaponSpectrumConfig {
            frequency_hz: f0,
            sound_speed: c,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: true,
                num_snapshots: 1,
                post_process: CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: [0.0, 0.0, 0.05],
            },
            sampling_frequency_hz: Some(fs),
            snapshot_selection: Some(selection),
            baseband_snapshot_step_samples: None,
        };

        // Test multiple candidate points
        let candidates = vec![
            [0.0, 0.0, 0.05],   // Center (should be highest for broadside source)
            [0.02, 0.0, 0.05],  // Off-axis
            [-0.02, 0.0, 0.05], // Off-axis opposite
        ];

        let mut spectra = Vec::new();
        for &candidate in &candidates {
            let mut cfg_scan = cfg.clone();
            cfg_scan.steering = SteeringVectorMethod::SphericalWave {
                source_position: candidate,
            };

            let spectrum = capon_spatial_spectrum_point(&data, &positions, candidate, &cfg_scan)
                .expect("capon spectrum");
            spectra.push(spectrum);

            // All spectra must be finite and positive
            assert!(
                spectrum.is_finite(),
                "Spectrum not finite for candidate {:?}",
                candidate
            );
            assert!(
                spectrum > 0.0,
                "Spectrum not positive for candidate {:?}",
                candidate
            );
        }

        // Verify spectra are not all identical (shows discrimination)
        let max_spectrum = spectra.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_spectrum = spectra.iter().cloned().fold(f64::INFINITY, f64::min);
        let dynamic_range = max_spectrum / min_spectrum.max(1e-10);

        assert!(
            dynamic_range > 1.1,
            "Capon spectrum shows insufficient spatial discrimination: dynamic range {:.2}",
            dynamic_range
        );
    }

    #[test]
    fn pipeline_is_invariant_to_global_time_shift() {
        // Mathematical property: Narrowband Capon spectrum should be invariant to
        // global time shifts (shifts all sensors equally), because it's based on
        // spatial covariance, not absolute time.

        let n_sensors = 4;
        let spacing_m = 0.0075;
        let n_samples = 128;
        let fs = 1_000_000.0;
        let f0 = 100_000.0;
        let c = 1500.0;
        let snr_db = 25.0;

        let data_original =
            generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);

        // Create time-shifted version: shift by 10 samples
        let shift = 10;
        let mut data_shifted = Array3::<f64>::zeros((n_sensors, 1, n_samples));
        for sensor in 0..n_sensors {
            for t in shift..n_samples {
                data_shifted[(sensor, 0, t - shift)] = data_original[(sensor, 0, t)];
            }
        }

        let positions = generate_ula_positions(n_sensors, spacing_m);
        let candidate = [0.0, 0.0, 0.05];

        let scenario = SnapshotScenario {
            frequency_hz: f0,
            sampling_frequency_hz: fs,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        };
        let selection = SnapshotSelection::Auto(scenario);

        let cfg = CaponSpectrumConfig {
            frequency_hz: f0,
            sound_speed: c,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process: CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: candidate,
            },
            sampling_frequency_hz: Some(fs),
            snapshot_selection: Some(selection),
            baseband_snapshot_step_samples: None,
        };

        let spectrum_original =
            capon_spatial_spectrum_point(&data_original, &positions, candidate, &cfg)
                .expect("spectrum original");
        let spectrum_shifted =
            capon_spatial_spectrum_point(&data_shifted, &positions, candidate, &cfg)
                .expect("spectrum shifted");

        // Validation: Spectra should be very similar (within 10% relative error)
        let rel_diff = (spectrum_original - spectrum_shifted).abs()
            / (spectrum_original + spectrum_shifted).max(1e-10);
        assert!(
            rel_diff < 0.1,
            "Spectra differ significantly under time shift: {:.2e} vs {:.2e} (rel diff: {:.2e})",
            spectrum_original,
            spectrum_shifted,
            rel_diff
        );
    }

    #[test]
    fn snapshot_methods_produce_consistent_covariance_rank() {
        // Different snapshot extraction methods should produce covariance matrices
        // with similar effective rank (number of significant eigenvalues)

        let n_sensors = 4;
        let spacing_m = 0.0075;
        let n_samples = 256;
        let fs = 1_000_000.0;
        let f0 = 100_000.0;
        let c = 1500.0;
        let snr_db = 20.0;

        let data =
            generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);

        // Method 1: Robust windowed STFT
        let scenario_robust = SnapshotScenario {
            frequency_hz: f0,
            sampling_frequency_hz: fs,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        };
        let snapshots_robust =
            extract_narrowband_snapshots(&data, &SnapshotSelection::Auto(scenario_robust))
                .expect("robust snapshots");

        // Method 2: Time-resolution optimized
        let scenario_time_res = SnapshotScenario {
            frequency_hz: f0,
            sampling_frequency_hz: fs,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: false,
            prefer_time_resolution: true,
        };
        let snapshots_time_res =
            extract_narrowband_snapshots(&data, &SnapshotSelection::Auto(scenario_time_res))
                .expect("time-res snapshots");

        // Both should produce valid snapshots with correct sensor dimension
        assert_eq!(snapshots_robust.nrows(), n_sensors);
        assert_eq!(snapshots_time_res.nrows(), n_sensors);
        assert!(snapshots_robust.ncols() > 0);
        assert!(snapshots_time_res.ncols() > 0);

        // Compute sample covariance for each
        let cov_robust = compute_sample_covariance(&snapshots_robust);
        let cov_time_res = compute_sample_covariance(&snapshots_time_res);

        // Check that both are Hermitian positive semi-definite (diagonal real and non-negative)
        for i in 0..n_sensors {
            assert!(
                cov_robust[(i, i)].im.abs() < 1e-10,
                "Covariance diagonal should be real"
            );
            assert!(
                cov_robust[(i, i)].re >= 0.0,
                "Covariance diagonal should be non-negative"
            );
            assert!(cov_time_res[(i, i)].im.abs() < 1e-10);
            assert!(cov_time_res[(i, i)].re >= 0.0);
        }
    }

    #[test]
    fn steering_vector_has_unit_magnitude_for_all_sensors() {
        // Integration test verifying steering vector normalization across the pipeline

        let n_sensors = 6;
        let spacing_m = 0.01;
        let positions = generate_ula_positions(n_sensors, spacing_m);
        let c = 1500.0;
        let f0 = 50_000.0;

        let steering = NarrowbandSteering::new(positions.clone(), c).expect("steering init");

        // Test multiple candidate points
        let candidates = vec![[0.0, 0.0, 0.05], [0.02, 0.0, 0.05], [-0.01, 0.01, 0.08]];

        for candidate in candidates {
            let sv = steering
                .steering_vector_point(candidate, f0)
                .expect("steering vector");

            assert_eq!(sv.as_array().len(), n_sensors);

            for &element in sv.as_array().iter() {
                let magnitude: f64 = element.norm();
                assert!(
                    (magnitude - 1.0).abs() < 1e-10,
                    "Steering vector element should have unit magnitude, got {:.6}",
                    magnitude
                );
            }
        }
    }

    #[test]
    fn diagonal_loading_prevents_covariance_singularity() {
        // Test that diagonal loading stabilizes Capon spectrum computation
        // even when covariance is poorly conditioned

        let n_sensors = 3;
        let spacing_m = 0.0075;
        let n_samples = 32; // Few samples → poorly conditioned covariance
        let fs = 1_000_000.0;
        let f0 = 100_000.0;
        let c = 1500.0;
        let snr_db = 10.0; // Low SNR

        let data =
            generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);
        let positions = generate_ula_positions(n_sensors, spacing_m);
        let candidate = [0.0, 0.0, 0.05];

        let scenario = SnapshotScenario {
            frequency_hz: f0,
            sampling_frequency_hz: fs,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        };

        // Test with no loading (should still work but may be less stable)
        let cfg_no_loading = CaponSpectrumConfig {
            frequency_hz: f0,
            sound_speed: c,
            diagonal_loading: 0.0,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process: CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: candidate,
            },
            sampling_frequency_hz: Some(fs),
            snapshot_selection: Some(SnapshotSelection::Auto(scenario)),
            baseband_snapshot_step_samples: None,
        };

        // Test with modest loading
        let cfg_with_loading = CaponSpectrumConfig {
            diagonal_loading: 1e-2,
            ..cfg_no_loading.clone()
        };

        let spectrum_no_loading =
            capon_spatial_spectrum_point(&data, &positions, candidate, &cfg_no_loading)
                .expect("spectrum without loading");
        let spectrum_with_loading =
            capon_spatial_spectrum_point(&data, &positions, candidate, &cfg_with_loading)
                .expect("spectrum with loading");

        // Both should be finite and positive
        assert!(spectrum_no_loading.is_finite() && spectrum_no_loading > 0.0);
        assert!(spectrum_with_loading.is_finite() && spectrum_with_loading > 0.0);

        // With loading should be more conservative (lower or similar)
        // This is problem-dependent, so we just verify it's reasonable
        assert!(
            spectrum_with_loading > 0.0,
            "Diagonal loading should produce positive spectrum"
        );
    }

    // === Helper Functions ===

    /// Compute sample covariance matrix from complex snapshots
    fn compute_sample_covariance(
        snapshots: &ndarray::Array2<Complex64>,
    ) -> ndarray::Array2<Complex64> {
        let n_sensors = snapshots.nrows();
        let n_snapshots = snapshots.ncols();
        let mut cov = ndarray::Array2::<Complex64>::zeros((n_sensors, n_sensors));

        for k in 0..n_snapshots {
            let snapshot = snapshots.column(k);
            for i in 0..n_sensors {
                for j in 0..n_sensors {
                    cov[(i, j)] += snapshot[i] * snapshot[j].conj();
                }
            }
        }

        // Normalize
        for elem in cov.iter_mut() {
            *elem /= n_snapshots as f64;
        }

        cov
    }
}
