use super::is_hermitian;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use num_complex::Complex64;

/// Estimate sample covariance matrix from multi-snapshot sensor data.
///
/// Computes the standard sample covariance estimator with optional diagonal loading
/// for numerical stability.
///
/// # Mathematical Definition
///
/// ```text
/// R = (1/M) ∑ₘ₌₁ᴹ x[m] x[m]^H + ε·I
/// ```
///
/// # Errors
///
/// Returns `Err(...)` if data has no snapshots, contains non-finite values,
/// `diagonal_loading` is negative/non-finite, or snapshots are insufficient for
/// full-rank estimation without loading.
pub fn estimate_sample_covariance(
    data: &Array2<Complex64>,
    diagonal_loading: f64,
) -> KwaversResult<Array2<Complex64>> {
    let (n_sensors, n_snapshots) = (data.nrows(), data.ncols());

    if n_snapshots == 0 {
        return Err(KwaversError::InvalidInput(
            "Covariance estimation requires at least one snapshot".into(),
        ));
    }

    if !diagonal_loading.is_finite() || diagonal_loading < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Diagonal loading must be non-negative and finite, got {}",
            diagonal_loading
        )));
    }

    if !data.iter().all(|&x| x.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "Input data contains non-finite values (NaN or Inf)".into(),
        ));
    }

    if n_snapshots < 2 * n_sensors && diagonal_loading == 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Insufficient snapshots for stable covariance estimation: {} snapshots for {} sensors \
             (recommend M ≥ 2N = {}). Use diagonal loading > 0.",
            n_snapshots,
            n_sensors,
            2 * n_sensors
        )));
    }

    // R = (1/M) * X * X^H
    let mut covariance = Array2::<Complex64>::zeros((n_sensors, n_sensors));
    for m in 0..n_snapshots {
        let snapshot = data.column(m);
        for i in 0..n_sensors {
            for j in 0..n_sensors {
                covariance[[i, j]] += snapshot[i] * snapshot[j].conj();
            }
        }
    }

    let scale = 1.0 / (n_snapshots as f64);
    covariance.mapv_inplace(|x| x * scale);

    if diagonal_loading > 0.0 {
        for i in 0..n_sensors {
            covariance[[i, i]] += Complex64::new(diagonal_loading, 0.0);
        }
    }

    if !is_hermitian(&covariance, 1e-10) {
        return Err(KwaversError::InvalidInput(
            "Computed covariance matrix is not Hermitian (numerical instability)".into(),
        ));
    }

    Ok(covariance)
}

/// Estimate covariance matrix using forward-backward averaging.
///
/// Forward-backward averaging doubles the effective number of snapshots and
/// enforces centro-Hermitian structure, improving estimation with limited data.
///
/// # Mathematical Definition
///
/// ```text
/// R_fb = (1/2) [R_f + J R_b^* J]
/// ```
///
/// where J is the exchange matrix (anti-diagonal identity).
///
/// Applicable to linear arrays and planar arrays with symmetric geometry.
///
/// # Errors
///
/// Same conditions as `estimate_sample_covariance`.
pub fn estimate_forward_backward_covariance(
    data: &Array2<Complex64>,
    diagonal_loading: f64,
) -> KwaversResult<Array2<Complex64>> {
    let r_forward = estimate_sample_covariance(data, 0.0)?;
    let n = data.nrows();

    // R_b = J R_f^* J: reverse rows and columns, conjugate
    let mut r_backward = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            r_backward[[i, j]] = r_forward[[n - 1 - i, n - 1 - j]].conj();
        }
    }

    // R_fb = (1/2) [R_f + R_b]
    let mut r_fb = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            r_fb[[i, j]] = (r_forward[[i, j]] + r_backward[[i, j]]) * 0.5;
        }
    }

    if diagonal_loading > 0.0 {
        for i in 0..n {
            r_fb[[i, i]] += Complex64::new(diagonal_loading, 0.0);
        }
    }

    if !is_hermitian(&r_fb, 1e-10) {
        return Err(KwaversError::InvalidInput(
            "Forward-backward covariance is not Hermitian (numerical instability)".into(),
        ));
    }

    Ok(r_fb)
}
