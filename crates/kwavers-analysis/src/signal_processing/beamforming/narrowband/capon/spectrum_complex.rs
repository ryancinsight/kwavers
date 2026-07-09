//! Complex-baseband Capon/MVDR spatial spectrum computation.

use super::CaponSpectrumConfig;
use crate::signal_processing::beamforming::narrowband::snapshots::{
    extract_complex_baseband_snapshots, extract_narrowband_snapshots, BasebandSnapshotConfig,
    SnapshotSelection,
};
use crate::signal_processing::beamforming::narrowband::steering::NarrowbandSteering;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::linear_algebra::ComplexLinearAlgebra;
use leto::Array3;
use eunomia::Complex64;

/// Compute the narrowband Capon/MVDR spatial spectrum using complex snapshots and Hermitian covariance.
///
/// `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`
///
/// Snapshot formation is controlled by `cfg.snapshot_selection`. When `None`, a conservative scenario
/// is auto-derived; on failure the legacy analytic-baseband model is used as a deterministic fallback.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
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
            "capon_spatial_spectrum_point_complex_baseband requires n_sensors > 0 and n_samples > 0".to_owned(),
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
            "capon_spatial_spectrum_point_complex_baseband: candidate must be finite".to_owned(),
        ));
    }

    let sampling_frequency_hz = cfg.sampling_frequency_hz.ok_or_else(|| {
        KwaversError::InvalidInput(
            "capon_spatial_spectrum_point_complex_baseband: cfg.sampling_frequency_hz is required"
                .to_owned(),
        )
    })?;

    // 1) Form complex snapshots.
    let auto_derived = cfg.snapshot_selection.is_none();

    let selection = cfg
        .snapshot_selection
        .clone()
        .unwrap_or(SnapshotSelection::Auto(
            crate::signal_processing::beamforming::narrowband::snapshots::SnapshotScenario {
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
                return Err(primary_err);
            }
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

    // 2) Estimate Hermitian covariance (complex).
    let mut r = cfg.covariance.estimate_complex(&x)?;

    // 3) Diagonal loading.
    if cfg.diagonal_loading > 0.0 {
        for i in 0..n_sensors {
            r[(i, i)] += Complex64::new(cfg.diagonal_loading, 0.0);
        }
    }

    // 4) Steering vector a(p): phase-only exp(-j 2π f τ(p)).
    let steering = NarrowbandSteering::new(sensor_positions.to_vec(), cfg.sound_speed)?;
    let a = steering
        .steering_vector_point(candidate, cfg.frequency_hz)?
        .into_array();

    // 5) Compute denom = aᴴ R^{-1} a via linear solve R y = a.
    let y = ComplexLinearAlgebra::solve_linear_system_complex(&r, &a)?;

    let mut denom = Complex64::new(0.0, 0.0);
    for i in 0..n_sensors {
        denom += a[i].conj() * y[i];
    }

    let denom_re = denom.re;
    if !denom_re.is_finite() || denom_re <= 1e-18 {
        return Err(KwaversError::Numerical(
            kwavers_core::error::NumericalError::InvalidOperation(
                "capon_spatial_spectrum_point_complex_baseband: non-positive or non-finite MVDR denominator".to_owned(),
            ),
        ));
    }

    Ok(1.0 / denom_re)
}
