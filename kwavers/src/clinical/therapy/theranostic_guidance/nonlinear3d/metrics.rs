//! Value-semantic 3-D reconstruction metrics.

use ndarray::Array3;

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
        nrmse: nrmse(&scores, &truth),
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

fn dice_equal_area(score: &[f64], target: &[bool]) -> f64 {
    let count = target.iter().filter(|active| **active).count();
    if count == 0 || score.is_empty() {
        return 0.0;
    }
    let mut ordered = score.to_vec();
    ordered.sort_by(f64::total_cmp);
    let threshold = ordered[ordered.len().saturating_sub(count)];
    let mut true_positive = 0usize;
    let mut predicted = 0usize;
    for (value, truth) in score.iter().zip(target.iter()) {
        let active = *value >= threshold;
        predicted += usize::from(active);
        true_positive += usize::from(active && *truth);
    }
    2.0 * true_positive as f64 / (predicted + count).max(1) as f64
}

fn cnr(score: &[f64], target: &[bool]) -> f64 {
    let lesion = score
        .iter()
        .zip(target.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .collect::<Vec<_>>();
    let background = score
        .iter()
        .zip(target.iter())
        .filter_map(|(value, active)| (!active).then_some(*value))
        .collect::<Vec<_>>();
    if lesion.is_empty() || background.len() < 2 {
        return 0.0;
    }
    let lesion_mean = mean(&lesion);
    let background_mean = mean(&background);
    let background_var = background
        .iter()
        .map(|value| (value - background_mean).powi(2))
        .sum::<f64>()
        / background.len() as f64;
    (lesion_mean - background_mean) / background_var.sqrt().max(1.0e-12)
}

fn nrmse(score: &[f64], target: &[bool]) -> f64 {
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

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}
