//! Value-semantic 3-D reconstruction metrics.
//!
//! `dice_equal_area` and `cnr` are the canonical implementations in
//! [`super::super::metrics`]; this module delegates to them to maintain
//! SSOT and avoid behavioural drift between the 2-D and 3-D metric paths.

use ndarray::Array3;

use super::super::metrics::{cnr, dice_equal_area};
use super::types::VolumeReconstructionMetrics;

pub(crate) fn metrics_from_score(
    score: &[f64],
    target: &[bool],
    body: &[bool],
) -> VolumeReconstructionMetrics {
    let selected = score
        .iter()
        .zip(target.iter())
        .zip(body.iter())
        .filter_map(|((s, t), b)| b.then_some((*s, *t)))
        .collect::<Vec<_>>();
    let scores = selected.iter().map(|(s, _)| *s).collect::<Vec<_>>();
    let truth = selected.iter().map(|(_, t)| *t).collect::<Vec<_>>();
    VolumeReconstructionMetrics {
        dice_equal_area: dice_equal_area(&scores, &truth),
        cnr: cnr(&scores, &truth),
        nrmse: binary_rmse(&scores, &truth),
    }
}

pub(crate) fn fused_score(
    fwi_score: &Array3<f64>,
    cavitation_density: &Array3<f64>,
    body: &Array3<bool>,
) -> Array3<f64> {
    let peak = cavitation_density
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .fold(0.0, f64::max)
        .max(1.0e-12);
    Array3::from_shape_fn(fwi_score.dim(), |idx| {
        if !body[idx] {
            return 0.0;
        }
        let fwi = fwi_score[idx].clamp(0.0, 1.0);
        let cav = (cavitation_density[idx] / peak).clamp(0.0, 1.0);
        (0.85 * fwi + 0.15 * cav).clamp(0.0, 1.0)
    })
}

/// Binary RMSE of a continuous score against a Boolean target mask.
///
/// Differs from the continuous `nrmse` in [`super::super::metrics`]: here the
/// reference signal is `{0, 1}` rather than a second continuous array, so no
/// range normalisation is meaningful — the denominator is simply `n`.
fn binary_rmse(score: &[f64], target: &[bool]) -> f64 {
    if score.is_empty() {
        return 0.0;
    }
    let mse = score
        .iter()
        .zip(target.iter())
        .map(|(value, active)| {
            let truth = if *active { 1.0 } else { 0.0 };
            (value.clamp(0.0, 1.0) - truth).powi(2)
        })
        .sum::<f64>()
        / score.len() as f64;
    mse.sqrt()
}
