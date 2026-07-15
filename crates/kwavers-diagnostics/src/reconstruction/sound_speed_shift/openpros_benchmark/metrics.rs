//! Reconstruction comparison metrics.

use leto::Array2;

use super::super::{SoundSpeedShiftImage, SoundSpeedShiftPlan};
use super::types::OpenProsShiftReconstructionMetrics;

pub(super) fn metrics_for(
    image: &SoundSpeedShiftImage,
    truth_shift_m_s: &Array2<f64>,
    active_mask: &Array2<bool>,
    plan: &SoundSpeedShiftPlan,
) -> OpenProsShiftReconstructionMetrics {
    let (mae, rmse, nrmse, pearson) =
        image_error_metrics(&image.sound_speed_shift_m_s, truth_shift_m_s, active_mask);
    let objective_initial = *image.objective_history.first().unwrap_or(&0.0);
    let objective_final = *image.objective_history.last().unwrap_or(&objective_initial);
    let objective_reduction_fraction = if objective_initial > f64::EPSILON {
        ((objective_initial - objective_final) / objective_initial).max(0.0)
    } else {
        0.0
    };

    OpenProsShiftReconstructionMetrics {
        rows_available: plan.rows_available(),
        rows_used: plan.rows_used(),
        active_voxels: plan.active_voxels(),
        stored_weight_count: plan.stored_weight_count(),
        mean_absolute_error_m_s: mae,
        root_mean_square_error_m_s: rmse,
        normalized_root_mean_square_error: nrmse,
        pearson_correlation: pearson,
        objective_initial,
        objective_final,
        objective_reduction_fraction,
    }
}

fn image_error_metrics(
    reconstructed: &Array2<f64>,
    truth: &Array2<f64>,
    active_mask: &Array2<bool>,
) -> (f64, f64, f64, f64) {
    debug_assert_eq!(reconstructed.shape(), truth.shape());
    debug_assert_eq!(reconstructed.shape(), active_mask.shape());

    let mut count = 0.0;
    let mut sum_abs = 0.0;
    let mut sum_sq = 0.0;
    let mut truth_sum_sq = 0.0;
    let mut recon_sum = 0.0;
    let mut truth_sum = 0.0;
    for ((idx, recon), truth_value) in reconstructed.indexed_iter().zip(truth.iter()) {
        if !active_mask[idx] {
            continue;
        }
        let diff = *recon - *truth_value;
        count += 1.0;
        sum_abs += diff.abs();
        sum_sq += diff * diff;
        truth_sum_sq += truth_value * truth_value;
        recon_sum += *recon;
        truth_sum += *truth_value;
    }

    let mae = sum_abs / count;
    let rmse = (sum_sq / count).sqrt();
    let nrmse = rmse / (truth_sum_sq / count).sqrt().max(f64::EPSILON);
    let recon_mean = recon_sum / count;
    let truth_mean = truth_sum / count;
    let mut covariance = 0.0;
    let mut recon_var = 0.0;
    let mut truth_var = 0.0;
    for ((idx, recon), truth_value) in reconstructed.indexed_iter().zip(truth.iter()) {
        if !active_mask[idx] {
            continue;
        }
        let centered_recon = *recon - recon_mean;
        let centered_truth = *truth_value - truth_mean;
        covariance += centered_recon * centered_truth;
        recon_var += centered_recon * centered_recon;
        truth_var += centered_truth * centered_truth;
    }
    let pearson = covariance / (recon_var * truth_var).sqrt().max(f64::EPSILON);
    (mae, rmse, nrmse, pearson)
}
