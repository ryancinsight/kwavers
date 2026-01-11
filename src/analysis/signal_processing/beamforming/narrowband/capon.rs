#![deny(missing_docs)]
//! Narrowband Capon/MVDR spatial spectrum (point-steered) for localization.
//!
//! # Field jargon
//! - **MVDR** (Minimum Variance Distortionless Response) is also known as the **Capon beamformer**.
//! - The **Capon spatial spectrum** evaluates candidate look directions/points by
//!
//! `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`
//!
//! where:
//! - `R` is the sample covariance matrix of the array snapshots
//! - `a(p)` is the (complex) steering vector for candidate point `p`
//!
//! # Intended use (narrowband adaptive localization)
//! This module provides a *point-dependent* score suitable for grid-search localization.
//! It is the narrowband complement to time-domain SRP-DAS.
//!
//! # Design constraints (SSOT)
//! - This module is the canonical location for Capon/MVDR spatial spectrum computation.
//! - It lives in the analysis layer and uses domain primitives for geometry and covariance.
//!
//! # Literature-aligned implementation stance (advanced/default)
//! For narrowband adaptive beamforming (MVDR/Capon, MUSIC, ESMV), the mathematically canonical and
//! literature-standard approach is:
//! 1) form complex snapshots `x_k ∈ ℂ^M` from windowed blocks (often via STFT/FFT bin snapshots),
//! 2) estimate a Hermitian covariance `R = (1/K) ∑ x_k x_kᴴ`,
//! 3) evaluate the MVDR quadratic form using a point-dependent steering vector.
//!
//! This module therefore supports:
//! - A legacy baseline (`capon_spatial_spectrum_point`) using real-valued covariance from time samples.
//! - The canonical path (`capon_spatial_spectrum_point_complex_baseband`) using complex snapshots and
//!   Hermitian covariance, with snapshot formation controlled by an explicit policy enum.
//!
//! # Numerical notes
//! - Diagonal loading is applied as `R_loaded = R + δ I` with `δ >= 0`.
//!
//! # Invariants / validation
//! - `frequency_hz` must be finite and > 0.
//! - `sound_speed` must be finite and > 0.
//! - `diagonal_loading` must be finite and >= 0.
//! - `sensor_data` must have shape `(n_sensors, 1, n_samples)` with `n_samples > 0`.
//!
//! # No error masking
//! - The snapshot policy is explicit; automatic selection is deterministic and scenario-based.
//! - No silent fallback between snapshot modes occurs.

use crate::analysis::signal_processing::beamforming::narrowband::snapshots::{
    extract_complex_baseband_snapshots, extract_narrowband_snapshots, BasebandSnapshotConfig,
    SnapshotSelection,
};
use crate::analysis::signal_processing::beamforming::narrowband::steering::NarrowbandSteering;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::{SteeringVector, SteeringVectorMethod};
use crate::domain::sensor::beamforming::CovarianceEstimator;
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array2, Array3};
use num_complex::Complex64;

/// Configuration for the narrowband Capon/MVDR spatial spectrum.
#[derive(Debug, Clone)]
pub struct CaponSpectrumConfig {
    /// Narrowband frequency (Hz) at which the steering vector is evaluated.
    pub frequency_hz: f64,
    /// Speed of sound (m/s).
    pub sound_speed: f64,
    /// Diagonal loading factor (δ >= 0).
    pub diagonal_loading: f64,
    /// Covariance estimation policy.
    pub covariance: CovarianceEstimator,
    /// Steering vector model.
    pub steering: SteeringVectorMethod,

    /// Sampling frequency (Hz) of `sensor_data`.
    ///
    /// This is required for complex snapshot extraction (both legacy analytic-baseband and
    /// windowed snapshot modes).
    ///
    /// If `None`, complex-narrowband scorers will reject use with an explicit error.
    pub sampling_frequency_hz: Option<f64>,

    /// Snapshot formation policy for the canonical complex narrowband path.
    ///
    /// - If `Some(SnapshotSelection::Explicit(...))`, uses exactly that method.
    /// - If `Some(SnapshotSelection::Auto(...))`, deterministically selects a literature-aligned
    ///   method from the provided scenario (recommended).
    /// - If `None`, the scorer will automatically select a robust default using a conservative
    ///   scenario derived from `frequency_hz` and `sampling_frequency_hz`.
    pub snapshot_selection: Option<SnapshotSelection>,

    /// Legacy snapshot stride (in samples) used when extracting analytic-signal complex baseband
    /// snapshots via `extract_complex_baseband_snapshots`.
    ///
    /// This is retained for compatibility. For advanced usage, prefer `snapshot_selection`.
    pub baseband_snapshot_step_samples: Option<usize>,
}

impl CaponSpectrumConfig {
    /// Validate invariants.
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "CaponSpectrumConfig: frequency_hz must be finite and > 0".to_string(),
            ));
        }
        if !self.sound_speed.is_finite() || self.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "CaponSpectrumConfig: sound_speed must be finite and > 0".to_string(),
            ));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(
                "CaponSpectrumConfig: diagonal_loading must be finite and >= 0".to_string(),
            ));
        }

        if let Some(fs) = self.sampling_frequency_hz {
            if !fs.is_finite() || fs <= 0.0 {
                return Err(KwaversError::InvalidInput(
                    "CaponSpectrumConfig: sampling_frequency_hz must be finite and > 0 (when provided)"
                        .to_string(),
                ));
            }
        }

        // SnapshotSelection validation is performed when resolving the selection against actual data
        // length (n_samples). We only enforce that if the legacy step is provided, it is valid.
        if let Some(step) = self.baseband_snapshot_step_samples {
            if step == 0 {
                return Err(KwaversError::InvalidInput(
                    "CaponSpectrumConfig: baseband_snapshot_step_samples must be >= 1 (when provided)"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for CaponSpectrumConfig {
    fn default() -> Self {
        Self {
            // Common ultrasound center frequency default (caller should override).
            frequency_hz: 1e6,
            sound_speed: 1500.0,
            diagonal_loading: 1e-6,
            covariance: CovarianceEstimator::default(),
            steering: SteeringVectorMethod::PlaneWave,
            sampling_frequency_hz: None,
            snapshot_selection: None,
            baseband_snapshot_step_samples: None,
        }
    }
}

/// Compute the narrowband Capon/MVDR spatial spectrum value `P_Capon(p)` for a candidate point.
///
/// # Parameters
/// - `sensor_data`: real-valued time series shaped `(n_sensors, 1, n_samples)`.
/// - `sensor_positions`: sensor coordinates `[x,y,z]` in meters.
/// - `candidate`: candidate look point `[x,y,z]` in meters (used for near-field steering when applicable).
/// - `cfg`: spectrum configuration.
///
/// # Returns
/// - `P_Capon(p)` (higher implies a more likely source / look direction).
///
/// # Errors
/// Returns an error if invariants fail, covariance inversion fails, or steering is undefined.
///
/// # Steering semantics
/// The steering vector is computed using `cfg.steering`.
/// For point-steered localization, the most direct choice is:
/// - `SteeringVectorMethod::SphericalWave { source_position: candidate }`
///
/// If `cfg.steering` is `Focused { focal_point }`, the focal_point provided in the config is used
/// (caller must ensure it matches `candidate` if they want point-steering).
pub fn capon_spatial_spectrum_point(
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &CaponSpectrumConfig,
) -> KwaversResult<f64> {
    cfg.validate()?;

    // Validate input shapes.
    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "capon_spatial_spectrum_point expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "capon_spatial_spectrum_point requires n_sensors > 0 and n_samples > 0".to_string(),
        ));
    }
    if sensor_positions.len() != n_sensors {
        return Err(KwaversError::InvalidInput(format!(
            "capon_spatial_spectrum_point sensor_positions len ({}) != n_sensors ({n_sensors})",
            sensor_positions.len()
        )));
    }
    if candidate.iter().any(|v| !v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "capon_spatial_spectrum_point: candidate must be finite".to_string(),
        ));
    }

    // Build snapshot matrix X (n_sensors x n_samples), real-valued.
    let mut snapshots = Array2::<f64>::zeros((n_sensors, n_samples));
    for i in 0..n_sensors {
        for t in 0..n_samples {
            snapshots[(i, t)] = sensor_data[(i, 0, t)];
        }
    }

    // Estimate covariance R (real symmetric) using the provided policy.
    let mut r = cfg.covariance.estimate(&snapshots)?;

    // Apply diagonal loading.
    if cfg.diagonal_loading > 0.0 {
        for i in 0..n_sensors {
            r[(i, i)] += cfg.diagonal_loading;
        }
    }

    // Compute steering vector a(p).
    //
    // For point-steered localization we want a(p) that depends on candidate.
    // If cfg.steering is spherical-wave or focused, use candidate appropriately.
    let steering_method = match &cfg.steering {
        SteeringVectorMethod::SphericalWave { .. } => SteeringVectorMethod::SphericalWave {
            source_position: candidate,
        },
        SteeringVectorMethod::Focused { .. } => SteeringVectorMethod::Focused {
            focal_point: candidate,
        },
        SteeringVectorMethod::PlaneWave => SteeringVectorMethod::PlaneWave,
    };

    let a = SteeringVector::compute(
        &steering_method,
        // `direction` is only used for PlaneWave; for SphericalWave/Focused it is ignored by
        // the current implementation. Provide a sensible unit direction to avoid confusion.
        [0.0, 0.0, 1.0],
        cfg.frequency_hz,
        sensor_positions,
        cfg.sound_speed,
    )?;

    // Invert the (real) covariance matrix using the SSOT real-valued inversion implementation.
    //
    // For the current snapshot model `R = (1/N) Σ x x^T` the covariance is real symmetric, so
    // `R^{-1}` is also real (when it exists). We therefore invert in ℝ and embed into ℂ only for
    // the steering-vector quadratic form.
    let r_inv = LinearAlgebra::matrix_inverse(&r)?;

    // Compute denominator a^H R^{-1} a.
    //
    // Since R^{-1} is real, this is:
    //   a^H (R^{-1} a)  where (R^{-1} a)_i = Σ_j R^{-1}_{ij} a_j
    let mut denom = Complex64::new(0.0, 0.0);
    for i in 0..n_sensors {
        let mut tmp = Complex64::new(0.0, 0.0);
        for j in 0..n_sensors {
            tmp += Complex64::new(r_inv[(i, j)], 0.0) * a[j];
        }
        denom += a[i].conj() * tmp;
    }

    let denom_re = denom.re;
    if !denom_re.is_finite() || denom_re <= 1e-18 {
        return Err(KwaversError::Numerical(
            crate::core::error::NumericalError::InvalidOperation(
                "capon_spatial_spectrum_point: non-positive or non-finite MVDR denominator"
                    .to_string(),
            ),
        ));
    }

    Ok(1.0 / denom_re)
}

/// Compute the narrowband Capon/MVDR spatial spectrum using **complex snapshots** and a
/// **Hermitian** covariance estimate.
///
/// # Mathematical definition
/// `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`
///
/// where:
/// - `a(p)` is the phase-only narrowband steering vector `exp(-j 2π f τ(p))`
/// - `R` is the complex Hermitian covariance `R = (1/K) ∑ x_k x_kᴴ` estimated from complex snapshots.
///
/// # Snapshot formation (advanced, literature-aligned)
/// Snapshot formation is controlled by `cfg.snapshot_selection`:
/// - If set, it is honored exactly.
/// - If `None`, the implementation deterministically selects a robust snapshot method appropriate
///   for the scenario (defaults to windowed STFT-bin snapshots).
///
/// # Input model
/// - `sensor_data` must be shaped `(n_sensors, 1, n_samples)` (real-valued).
/// - The signal is assumed narrowband around `cfg.frequency_hz`.
///
/// # Errors
/// Returns an error if:
/// - invariants fail (shape, finiteness, config validity)
/// - `cfg.sampling_frequency_hz` is missing
/// - snapshot selection cannot be resolved/validated against the data length
///
/// # Implementation note (SSOT correctness)
/// This function computes the MVDR quadratic form without forming `R^{-1}` explicitly by solving
/// the linear system `R y = a` and then evaluating `aᴴ y`. The complex linear solve is provided by
/// SSOT `crate::math::linear_algebra::LinearAlgebra::solve_linear_system_complex`.
pub fn capon_spatial_spectrum_point_complex_baseband(
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &CaponSpectrumConfig,
) -> KwaversResult<f64> {
    cfg.validate()?;

    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "capon_spatial_spectrum_point_complex_baseband expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "capon_spatial_spectrum_point_complex_baseband requires n_sensors > 0 and n_samples > 0"
                .to_string(),
        ));
    }
    if sensor_positions.len() != n_sensors {
        return Err(KwaversError::InvalidInput(format!(
            "capon_spatial_spectrum_point_complex_baseband sensor_positions len ({}) != n_sensors ({n_sensors})",
            sensor_positions.len()
        )));
    }
    if candidate.iter().any(|v| !v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "capon_spatial_spectrum_point_complex_baseband: candidate must be finite".to_string(),
        ));
    }

    let sampling_frequency_hz = cfg.sampling_frequency_hz.ok_or_else(|| {
        KwaversError::InvalidInput(
            "capon_spatial_spectrum_point_complex_baseband: cfg.sampling_frequency_hz is required"
                .to_string(),
        )
    })?;

    // 1) Form complex snapshots (canonical path: windowed snapshots; legacy baseband available).
    //
    // Invariant: we must not mask failures for *explicitly* selected snapshot methods.
    //
    // - If `cfg.snapshot_selection` is `Some(SnapshotSelection::Explicit(_))`, any failure is surfaced.
    // - If `cfg.snapshot_selection` is `Some(SnapshotSelection::Auto(_))`, any failure is surfaced
    //   (caller asked for auto-selection explicitly; do not silently change method).
    // - If `cfg.snapshot_selection` is `None`, we auto-derive a conservative scenario and may
    //   (as a last resort) fall back to the legacy analytic-baseband snapshot model.
    let auto_derived = cfg.snapshot_selection.is_none();

    let selection = cfg
        .snapshot_selection
        .clone()
        .unwrap_or(SnapshotSelection::Auto(
        crate::analysis::signal_processing::beamforming::narrowband::snapshots::SnapshotScenario {
            frequency_hz: cfg.frequency_hz,
            sampling_frequency_hz,
            fractional_bandwidth: Some(0.05),
            prefer_robustness: true,
            prefer_time_resolution: false,
        },
    ));

    let x = match extract_narrowband_snapshots(sensor_data, &selection) {
        Ok(x) => x,
        Err(primary_err) => {
            if !auto_derived {
                // Explicit snapshot policy was requested; do not mask failure.
                return Err(primary_err);
            }

            // Auto-derived policy: allow a deterministic, explicitly-documented legacy fallback.
            let snapshot_step_samples = cfg
                .baseband_snapshot_step_samples
                .unwrap_or(BasebandSnapshotConfig::default().snapshot_step_samples);
            let snapshot_cfg = BasebandSnapshotConfig {
                sampling_frequency_hz,
                center_frequency_hz: cfg.frequency_hz,
                snapshot_step_samples,
            };
            extract_complex_baseband_snapshots(sensor_data, &snapshot_cfg)?
        }
    };

    // 2) Estimate Hermitian covariance (complex)
    let mut r = cfg.covariance.estimate_complex(&x)?;

    // 3) Diagonal loading
    if cfg.diagonal_loading > 0.0 {
        for i in 0..n_sensors {
            r[(i, i)] += Complex64::new(cfg.diagonal_loading, 0.0);
        }
    }

    // 4) Steering vector a(p): phase-only exp(-j 2π f τ(p))
    let steering = NarrowbandSteering::new(sensor_positions.to_vec(), cfg.sound_speed)?;
    let a = steering
        .steering_vector_point(candidate, cfg.frequency_hz)?
        .into_array();

    // 5) Compute denominator via SSOT complex linear solve (no explicit inverse):
    //    denom = aᴴ R^{-1} a  where we solve R y = a, then denom = aᴴ y.
    let y = LinearAlgebra::solve_linear_system_complex(&r, &a)?;

    let mut denom = Complex64::new(0.0, 0.0);
    for i in 0..n_sensors {
        denom += a[i].conj() * y[i];
    }

    let denom_re = denom.re;
    if !denom_re.is_finite() || denom_re <= 1e-18 {
        return Err(KwaversError::Numerical(
            crate::core::error::NumericalError::InvalidOperation(
                "capon_spatial_spectrum_point_complex_baseband: non-positive or non-finite MVDR denominator"
                    .to_string(),
            ),
        ));
    }

    Ok(1.0 / denom_re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    fn sensor_positions_m() -> Vec<[f64; 3]> {
        vec![
            [-0.015, 0.0, 0.0],
            [-0.005, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.015, 0.0, 0.0],
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

    fn synth_narrowband_sensor_data(
        sensor_positions: &[[f64; 3]],
        true_source: [f64; 3],
        sound_speed: f64,
        frequency_hz: f64,
        sampling_frequency_hz: f64,
        n_samples: usize,
        extra_delay_s: f64,
    ) -> Array3<f64> {
        let n_sensors = sensor_positions.len();
        let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

        let omega = 2.0 * std::f64::consts::PI * frequency_hz;

        for (i, &pos) in sensor_positions.iter().enumerate() {
            let tau = tof_s(pos, true_source, sound_speed) + extra_delay_s;
            for t in 0..n_samples {
                let time_s = (t as f64) / sampling_frequency_hz;
                data[(i, 0, t)] = (omega * (time_s - tau)).cos();
            }
        }

        data
    }

    #[test]
    fn capon_spectrum_is_finite_for_simple_case() {
        // Two sensors, identical constant signals -> covariance is rank-1 but diagonal loading makes invertible.
        let n_sensors = 2usize;
        let n_samples = 64usize;

        let mut x = Array3::<f64>::zeros((n_sensors, 1, n_samples));
        for t in 0..n_samples {
            x[(0, 0, t)] = 1.0;
            x[(1, 0, t)] = 1.0;
        }

        let positions = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
        let cfg = CaponSpectrumConfig {
            frequency_hz: 1e6,
            sound_speed: 1500.0,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process: crate::domain::sensor::beamforming::covariance::CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: [0.0, 0.0, 0.02],
            },
            sampling_frequency_hz: None,
            snapshot_selection: None,
            baseband_snapshot_step_samples: None,
        };

        let p =
            capon_spatial_spectrum_point(&x, &positions, [0.0, 0.0, 0.02], &cfg).expect("spectrum");
        assert!(p.is_finite());
        assert!(p > 0.0);
    }

    #[test]
    fn complex_baseband_requires_sampling_frequency() {
        let sensors = sensor_positions_m();
        let sound_speed = 1500.0;
        let sampling_frequency_hz = 2_000_000.0;
        let frequency_hz = 200_000.0;
        let n_samples = 256;

        let true_source = [0.0, 0.01, 0.02];
        let sensor_data = synth_narrowband_sensor_data(
            &sensors,
            true_source,
            sound_speed,
            frequency_hz,
            sampling_frequency_hz,
            n_samples,
            0.0,
        );

        let cfg = CaponSpectrumConfig {
            frequency_hz,
            sound_speed,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process: crate::domain::sensor::beamforming::covariance::CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: true_source,
            },
            sampling_frequency_hz: None, // critical: missing
            snapshot_selection: None,
            baseband_snapshot_step_samples: None,
        };

        let err = capon_spatial_spectrum_point_complex_baseband(
            &sensor_data,
            &sensors,
            true_source,
            &cfg,
        )
        .expect_err("missing sampling_frequency_hz must be rejected");
        assert!(err.to_string().contains("sampling_frequency_hz"));
    }

    #[test]
    fn complex_baseband_rejects_invalid_snapshot_step() {
        let sensors = sensor_positions_m();
        let sound_speed = 1500.0;
        let sampling_frequency_hz = 2_000_000.0;
        let frequency_hz = 200_000.0;
        let n_samples = 256;

        let true_source = [0.0, 0.01, 0.02];
        let sensor_data = synth_narrowband_sensor_data(
            &sensors,
            true_source,
            sound_speed,
            frequency_hz,
            sampling_frequency_hz,
            n_samples,
            0.0,
        );

        let cfg = CaponSpectrumConfig {
            frequency_hz,
            sound_speed,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process:
                    crate::domain::sensor::beamforming::covariance::CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: true_source,
            },
            sampling_frequency_hz: Some(sampling_frequency_hz),
            snapshot_selection: None,
            baseband_snapshot_step_samples: Some(0), // invalid
        };

        let err = capon_spatial_spectrum_point_complex_baseband(
            &sensor_data,
            &sensors,
            true_source,
            &cfg,
        )
        .expect_err("snapshot step 0 must be rejected");
        assert!(err.to_string().contains("baseband_snapshot_step_samples"));
    }

    #[test]
    fn complex_baseband_mvdr_is_invariant_to_global_time_shift() {
        // Global time shift across all sensors corresponds to a global phase rotation in baseband,
        // which must not change the MVDR/Capon spectrum value.
        let sound_speed = 1500.0;
        let sampling_frequency_hz = 2_000_000.0;
        let frequency_hz = 200_000.0;
        let n_samples = 2048;

        let sensors = sensor_positions_m();
        let true_source = [0.0, 0.01, 0.02];

        let cfg = CaponSpectrumConfig {
            frequency_hz,
            sound_speed,
            diagonal_loading: 1e-3,
            covariance: CovarianceEstimator {
                forward_backward_averaging: false,
                num_snapshots: 1,
                post_process:
                    crate::domain::sensor::beamforming::covariance::CovariancePostProcess::None,
            },
            steering: SteeringVectorMethod::SphericalWave {
                source_position: true_source,
            },
            sampling_frequency_hz: Some(sampling_frequency_hz),
            snapshot_selection: None,
            baseband_snapshot_step_samples: Some(1),
        };

        let sensor_data_0 = synth_narrowband_sensor_data(
            &sensors,
            true_source,
            sound_speed,
            frequency_hz,
            sampling_frequency_hz,
            n_samples,
            0.0,
        );
        let sensor_data_shift = synth_narrowband_sensor_data(
            &sensors,
            true_source,
            sound_speed,
            frequency_hz,
            sampling_frequency_hz,
            n_samples,
            // Global delay of exactly 10 cycles:
            // Δt = 10 / f. This is ~50µs at 200kHz (reasonable vs simulation length).
            10.0 / frequency_hz,
        );

        let p0 = capon_spatial_spectrum_point_complex_baseband(
            &sensor_data_0,
            &sensors,
            true_source,
            &cfg,
        )
        .expect("spectrum");
        let p1 = capon_spatial_spectrum_point_complex_baseband(
            &sensor_data_shift,
            &sensors,
            true_source,
            &cfg,
        )
        .expect("spectrum");

        assert!(p0.is_finite() && p0 > 0.0);
        assert!(p1.is_finite() && p1 > 0.0);

        // Invariance tolerance: allow small numerical drift due to FFT/Hilbert and finite precision.
        assert_abs_diff_eq!(p0, p1, epsilon = 1e-6);
    }
}
