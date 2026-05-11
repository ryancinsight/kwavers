//! Real-valued covariance Capon/MVDR spatial spectrum computation.

use super::CaponSpectrumConfig;
use crate::analysis::signal_processing::beamforming::utils::steering::{
    SteeringVector, SteeringVectorMethod,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array2, Array3};
use num_complex::Complex64;

/// Compute the narrowband Capon/MVDR spatial spectrum value `P_Capon(p)` for a candidate point.
///
/// # Parameters
/// - `sensor_data`: real-valued time series shaped `(n_sensors, 1, n_samples)`.
/// - `sensor_positions`: sensor coordinates `[x,y,z]` in meters.
/// - `candidate`: candidate look point `[x,y,z]` in meters.
/// - `cfg`: spectrum configuration.
///
/// # Returns
/// `P_Capon(p)` — higher implies a more likely source / look direction.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn capon_spatial_spectrum_point(
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &CaponSpectrumConfig,
) -> KwaversResult<f64> {
    cfg.validate()?;

    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "capon_spatial_spectrum_point expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "capon_spatial_spectrum_point requires n_sensors > 0 and n_samples > 0".to_owned(),
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
            "capon_spatial_spectrum_point: candidate must be finite".to_owned(),
        ));
    }

    // Build snapshot matrix X (n_sensors x n_samples), real-valued.
    let mut snapshots = Array2::<f64>::zeros((n_sensors, n_samples));
    for i in 0..n_sensors {
        for t in 0..n_samples {
            snapshots[(i, t)] = sensor_data[(i, 0, t)];
        }
    }

    // Estimate covariance R (real symmetric).
    let mut r = cfg.covariance.estimate(&snapshots)?;

    if cfg.diagonal_loading > 0.0 {
        for i in 0..n_sensors {
            r[(i, i)] += cfg.diagonal_loading;
        }
    }

    // Build point-dependent steering vector.
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
        [0.0, 0.0, 1.0],
        cfg.frequency_hz,
        sensor_positions,
        cfg.sound_speed,
    )?;

    // Invert the real covariance matrix.
    let r_inv = LinearAlgebra::matrix_inverse(&r)?;

    // Compute denominator a^H R^{-1} a.
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
                "capon_spatial_spectrum_point: non-positive or non-finite MVDR denominator".to_owned(),
            ),
        ));
    }

    Ok(1.0 / denom_re)
}
