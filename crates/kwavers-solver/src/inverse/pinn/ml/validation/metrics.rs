//! Quantitative validation metrics: MAE, RMSE, relative L2, Pearson correlation.

use super::PinnValidationMetrics;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;

/// Compute MAE, RMSE, relative L2 error, and max pointwise error.
/// # Errors
/// - Returns [`crate::KwaversError::InvalidInput`] if reference and prediction have different shapes.
///
pub fn compute_validation_metrics(
    reference: &Array2<f64>,
    prediction: &Array2<f64>,
) -> KwaversResult<PinnValidationMetrics> {
    if reference.shape() != prediction.shape() {
        return Err(KwaversError::InvalidInput(
            "Reference and prediction must have same dimensions".to_owned(),
        ));
    }

    let [nx, nt] = reference.shape();
    let mut sum_abs_error = 0.0_f64;
    let mut sum_squared_error = 0.0_f64;
    let mut max_error = 0.0_f64;
    let mut sum_squared_ref = 0.0_f64;

    for i in 0..nx {
        for j in 0..nt {
            let ref_val = reference[[i, j]];
            let pred_val = prediction[[i, j]];
            let error = (pred_val - ref_val).abs();
            sum_abs_error += error;
            sum_squared_error += error * error;
            sum_squared_ref += ref_val * ref_val;
            max_error = max_error.max(error);
        }
    }

    let n = (nx * nt) as f64;
    let mean_absolute_error = sum_abs_error / n;
    let rmse = (sum_squared_error / n).sqrt();
    let relative_l2_error = if sum_squared_ref > 0.0 {
        (sum_squared_error / sum_squared_ref).sqrt()
    } else {
        0.0
    };

    Ok(PinnValidationMetrics {
        mean_absolute_error,
        rmse,
        relative_l2_error,
        max_error,
    })
}

/// Compute Pearson correlation coefficient between reference and prediction.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn compute_correlation(
    reference: &Array2<f64>,
    prediction: &Array2<f64>,
) -> KwaversResult<f64> {
    let n = (reference.len()) as f64;
    let mean_ref: f64 = reference.iter().sum::<f64>() / n;
    let mean_pred: f64 = prediction.iter().sum::<f64>() / n;

    let mut sum_prod = 0.0_f64;
    let mut sum_sq_ref = 0.0_f64;
    let mut sum_sq_pred = 0.0_f64;

    for (r, p) in reference.iter().zip(prediction.iter()) {
        let diff_ref = r - mean_ref;
        let diff_pred = p - mean_pred;
        sum_prod += diff_ref * diff_pred;
        sum_sq_ref += diff_ref * diff_ref;
        sum_sq_pred += diff_pred * diff_pred;
    }

    let correlation = if sum_sq_ref > 0.0 && sum_sq_pred > 0.0 {
        sum_prod / (sum_sq_ref.sqrt() * sum_sq_pred.sqrt())
    } else {
        0.0
    };

    Ok(correlation)
}

/// Compute mean relative error, skipping points where |reference| ≤ 1e-10.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn compute_mean_relative_error(
    reference: &Array2<f64>,
    prediction: &Array2<f64>,
) -> KwaversResult<f64> {
    let epsilon = 1e-10;
    let mut sum_relative_error = 0.0_f64;
    let mut count = 0usize;

    for (r, p) in reference.iter().zip(prediction.iter()) {
        if r.abs() > epsilon {
            sum_relative_error += ((p - r) / r).abs();
            count += 1;
        }
    }

    Ok(if count > 0 {
        sum_relative_error / count as f64
    } else {
        0.0
    })
}
