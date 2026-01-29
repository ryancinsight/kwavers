//! Release tests for narrowband point-steered Capon/MVDR spatial spectrum localization scorer.
//!
//! # Field jargon
//! - **MVDR** (Minimum Variance Distortionless Response) is also known as the **Capon beamformer**.
//! - The **Capon/MVDR spatial spectrum** is:
//!
//!   `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`
//!
//!   where `a(p)` is the (complex) steering vector at candidate point `p` and `R` is the sample
//!   covariance matrix (often with **diagonal loading** and/or **shrinkage** for robustness).
//!
//! # What this test enforces
//! - The scorer for method **B** is point-dependent and peaks near the true source for a narrowband
//!   propagation-delayed tone when using near-field steering.
//! - We do **not** re-implement MVDR math here; we use SSOT beamforming primitives:
//!   `kwavers::domain::sensor::beamforming::capon_spatial_spectrum_point_complex_baseband`.
//!
//! # Signal model (narrowband, real-valued)
//! We synthesize a narrowband real sinusoid observed at each sensor with a propagation delay:
//!
//! `x_i(t) = cos(2π f (t - τ_i(p_true)))`
//!
//! with `τ_i(p) = ||x_i - p|| / c`.
//!
//! # Snapshot/covariance model (advanced, literature-aligned)
//! The MVDR scorer uses the SSOT **complex-snapshot** path with Hermitian covariance
//! `R = (1/K) ∑ x_k x_kᴴ`.
//!
//! Snapshot formation is controlled by configuration (explicit policy enum). This test intentionally
//! exercises the “advanced default” behavior by leaving snapshot selection as `None`, allowing the
//! SSOT scorer to select a robust windowed snapshot method deterministically.

use kwavers::analysis::signal_processing::beamforming::covariance::{
    CovarianceEstimator, CovariancePostProcess,
};
use kwavers::analysis::signal_processing::beamforming::narrowband::capon::{
    capon_spatial_spectrum_point_complex_baseband, CaponSpectrumConfig,
};
use kwavers::analysis::signal_processing::beamforming::utils::steering::SteeringVectorMethod;
use kwavers::domain::sensor::Position;
use ndarray::Array3;

/// Construct a small 2D array (xy-plane, z=0) with modest aperture.
///
/// This matches the style used in other localization tests but keeps this file standalone.
fn sensor_positions_m() -> Vec<[f64; 3]> {
    // IMPORTANT (identifiability):
    // A purely 1D array lying on the x-axis is mirror-ambiguous in y for a single narrowband tone:
    // points (x, +y, z) and (x, -y, z) yield identical ranges to all sensors when all sensors have y=0,
    // so MVDR/Capon can legitimately peak at the mirrored location on a coarse grid.
    //
    // Break that symmetry by using a small 2D aperture in the xy-plane.
    vec![
        // left pair
        [-0.015, -0.005, 0.0],
        [-0.015, 0.005, 0.0],
        // right pair
        [0.015, -0.005, 0.0],
        [0.015, 0.005, 0.0],
    ]
}

fn euclidean_distance_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn tof_s(sensor_pos: [f64; 3], source_pos: [f64; 3], sound_speed: f64) -> f64 {
    euclidean_distance_m(sensor_pos, source_pos) / sound_speed
}

/// Synthesize a narrowband signal that is *consistent* with complex-baseband (analytic) processing.
///
/// The SSOT scorer path `capon_spatial_spectrum_point_complex_baseband` forms complex snapshots
/// around `cfg.frequency_hz`. Feeding a purely real cosine creates a conjugate-symmetric spectrum
/// (±f) and can produce ambiguous steering responses on coarse grids.
///
/// To remove this modeling inconsistency deterministically, we synthesize the **analytic** signal
/// as a real-valued time series containing both quadrature components:
///
/// `x_i(t) = cos(ω(t-τ_i)) + cos(ω(t-τ_i) + π/2)`
///
/// i.e. `cos(φ) - sin(φ)`; this is equivalent to a fixed complex phasor multiplier `(1 - j)` and
/// is consistent with forming complex baseband snapshots at `+f` only.
///
/// This is equivalent to injecting a fixed phase offset and breaks the ±f ambiguity without
/// introducing randomness/noise.
///
/// Output shape: `(n_sensors, 1, n_samples)`.
fn synth_narrowband_sensor_data(
    sensor_positions: &[[f64; 3]],
    true_source: [f64; 3],
    sound_speed: f64,
    frequency_hz: f64,
    sampling_frequency_hz: f64,
    n_samples: usize,
) -> Array3<f64> {
    let n_sensors = sensor_positions.len();
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

    let omega = 2.0 * std::f64::consts::PI * frequency_hz;

    // IMPORTANT:
    // For a deterministic, physically identifiable narrowband test signal compatible with the
    // complex-baseband SSOT path, we generate a quadrature-coded real series:
    //
    //   x(t) = cos(φ) + cos(φ + π/2) = cos(φ) - sin(φ)
    //
    // This is equivalent to multiplying the analytic signal exp(jφ) by (1 - j) and then taking
    // the real part, which ensures the +f baseband extraction sees a consistent complex phasor
    // (i.e., avoids the conjugate-symmetry ambiguity of a pure real cosine under coarse search).
    for (i, &pos) in sensor_positions.iter().enumerate() {
        let tau = tof_s(pos, true_source, sound_speed);
        for t in 0..n_samples {
            let time_s = (t as f64) / sampling_frequency_hz;
            let phase = omega * (time_s - tau);
            data[(i, 0, t)] = phase.cos() + (phase + std::f64::consts::FRAC_PI_2).cos();
        }
    }

    data
}

#[test]
fn capon_mvdr_spectrum_peaks_near_true_source_on_grid() {
    // Deterministic, modest sizes: release-test friendly.
    let sound_speed = 1500.0;
    let sampling_frequency_hz = 2_000_000.0; // 2 MHz
    let frequency_hz = 200_000.0; // 200 kHz narrowband tone (well below Nyquist)
    let n_samples = 2048;

    let sensors = sensor_positions_m();

    // True source within a small search region.
    let true_source = Position::new(0.0, 0.01, 0.02);
    let true_p = true_source.to_array();

    let sensor_data = synth_narrowband_sensor_data(
        &sensors,
        true_p,
        sound_speed,
        frequency_hz,
        sampling_frequency_hz,
        n_samples,
    );

    // Capon/MVDR spectrum configuration:
    // - Near-field point localization: spherical-wave steering.
    // - Diagonal loading for robustness (especially with coherent tones).
    let base_cfg = CaponSpectrumConfig {
        frequency_hz,
        sound_speed,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::default(),
        },
        // `capon_spatial_spectrum_point_complex_baseband` uses the `candidate` argument to build
        // the steering vector for the MVDR quadratic form. We still populate `cfg.steering` for
        // config completeness and compatibility with other paths.
        steering: SteeringVectorMethod::SphericalWave {
            source_position: true_p,
        },
        sampling_frequency_hz: Some(sampling_frequency_hz),
        // Advanced snapshot selection: leave this as None to exercise SSOT auto-derived defaults.
        snapshot_selection: None,
        // Legacy analytic-baseband fallback stride (only used if SSOT auto-derived selection
        // cannot be resolved; deterministic and documented).
        baseband_snapshot_step_samples: None,
    };

    // Grid: cube around the true source with 5 mm spacing.
    //
    // Rationale:
    // With only 4 sensors and a single-tone narrowband model, the MVDR spectrum can have multiple
    // nearby local maxima on a coarse grid (especially in z). Using a 5 mm step keeps the test
    // deterministic while providing enough resolution that the global maximum lands near the
    // physically correct point.

    let mut best_score = f64::NEG_INFINITY;
    let mut best_point = None::<[f64; 3]>;

    // Explicit grid coordinates (avoid floating drift from iterative accumulation).
    let xs = [
        -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
    ];
    let ys = xs;
    let zs = xs;

    // Center grid around true source.
    for dx in xs {
        for dy in ys {
            for dz in zs {
                let p = [true_p[0] + dx, true_p[1] + dy, true_p[2] + dz];
                // Ensure steering configuration is consistent with the candidate point.
                let cfg = CaponSpectrumConfig {
                    steering: SteeringVectorMethod::SphericalWave { source_position: p },
                    ..base_cfg.clone()
                };

                let score =
                    capon_spatial_spectrum_point_complex_baseband(&sensor_data, &sensors, p, &cfg)
                        .expect("Capon spectrum must be computable");

                if score > best_score {
                    best_score = score;
                    best_point = Some(p);
                }
            }
        }
    }

    let best_point = best_point.expect("grid must be non-empty");
    let estimated = Position::from_array(best_point);

    // With 5mm grid step, expect within a few steps due to finite-snapshot effects and diagonal loading.
    let err_m = estimated.distance_to(&true_source);
    assert!(
        err_m <= 0.018,
        "expected MVDR/Capon spectrum peak within 1.8cm; got err={err_m}m (estimated={estimated:?}, true={true_source:?}, best_score={best_score})"
    );

    assert!(best_score.is_finite());
    assert!(best_score > 0.0);
}

#[test]
fn capon_mvdr_spectrum_rejects_invalid_frequency() {
    let sound_speed = 1500.0;
    let sampling_frequency_hz = 1_000_000.0;
    let frequency_hz = 0.0; // invalid
    let n_samples = 256;

    let sensors = sensor_positions_m();
    let true_source = Position::new(0.0, 0.01, 0.02).to_array();

    let sensor_data = synth_narrowband_sensor_data(
        &sensors,
        true_source,
        sound_speed,
        100_000.0, // synth with a valid tone; config will be invalid instead
        sampling_frequency_hz,
        n_samples,
    );

    let cfg = CaponSpectrumConfig {
        frequency_hz,
        sound_speed,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::default(),
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: true_source,
        },
        sampling_frequency_hz: Some(sampling_frequency_hz),
        snapshot_selection: None,
        baseband_snapshot_step_samples: None,
    };

    let err =
        capon_spatial_spectrum_point_complex_baseband(&sensor_data, &sensors, true_source, &cfg)
            .expect_err("invalid frequency must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("frequency_hz") || msg.contains("frequency"),
        "expected error mentioning frequency; got: {msg}"
    );
}
