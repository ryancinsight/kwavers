//! Value-semantic metrics for theranostic inverse outputs.

use kwavers_math::statistics::{nrmse, pearson};
use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct ReconstructionMetrics {
    pub pearson: f64,
    pub nrmse: f64,
    pub dice_equal_area: f64,
    pub cnr: f64,
}

pub fn metrics_for(
    target: &Array2<f64>,
    recon: &Array2<f64>,
    mask: &Array2<bool>,
) -> ReconstructionMetrics {
    let t = masked_values(target, mask);
    let r = masked_values(recon, mask);
    ReconstructionMetrics {
        pearson: pearson(&t, &r),
        nrmse: nrmse(&t, &r),
        dice_equal_area: dice_equal_area(&r, &positive_mask(&t)),
        cnr: cnr(&r, &positive_mask(&t)),
    }
}

pub fn masked_values(image: &Array2<f64>, mask: &Array2<bool>) -> Vec<f64> {
    image
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .collect()
}

pub fn dice_equal_area(score: &[f64], target: &[bool]) -> f64 {
    if score.len() != target.len() || score.is_empty() {
        return 0.0;
    }
    let count = target.iter().filter(|v| **v).count();
    if count == 0 {
        return 0.0;
    }
    let (min_score, max_score) = score.iter().copied().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_v, max_v), value| (min_v.min(value), max_v.max(value)),
    );
    if !min_score.is_finite() || !max_score.is_finite() || min_score == max_score {
        return 0.0;
    }
    let mut ordered = score.to_vec();
    ordered.sort_by(f64::total_cmp);
    let threshold = ordered[ordered.len().saturating_sub(count)];
    let mut tp = 0usize;
    let mut pred = 0usize;
    for (value, truth) in score.iter().zip(target.iter()) {
        let selected = *value >= threshold;
        pred += usize::from(selected);
        tp += usize::from(selected && *truth);
    }
    2.0 * tp as f64 / (pred + count).max(1) as f64
}

pub fn cnr(score: &[f64], target: &[bool]) -> f64 {
    let mut lesion = Vec::new();
    let mut background = Vec::new();
    for (value, truth) in score.iter().zip(target.iter()) {
        if *truth {
            lesion.push(*value);
        } else {
            background.push(*value);
        }
    }
    if lesion.is_empty() || background.len() < 2 {
        return 0.0;
    }
    let ml = lesion.iter().sum::<f64>() / lesion.len() as f64;
    let mb = background.iter().sum::<f64>() / background.len() as f64;
    let vb = background.iter().map(|v| (*v - mb).powi(2)).sum::<f64>() / background.len() as f64;
    (ml - mb) / vb.sqrt().max(1.0e-12)
}

fn positive_mask(values: &[f64]) -> Vec<bool> {
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    values.iter().map(|v| *v >= 0.45 * max_value).collect()
}

#[cfg(test)]
mod tests {
    use super::dice_equal_area;

    #[test]
    fn dice_equal_area_rejects_flat_score_field() {
        let score = [0.0, 0.0, 0.0, 0.0];
        let target = [false, true, false, true];

        let dice = dice_equal_area(&score, &target);

        assert_eq!(dice, 0.0);
    }

    #[test]
    fn dice_equal_area_selects_equal_target_area_for_ranked_score() {
        let score = [0.1, 0.9, 0.2, 0.8];
        let target = [false, true, false, true];

        let dice = dice_equal_area(&score, &target);

        assert_eq!(dice, 1.0);
    }
}
