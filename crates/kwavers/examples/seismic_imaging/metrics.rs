//! Shared value-semantic metrics for the 2-D and 3-D seismic examples.

use leto::Array3;

/// Print whole-model error metrics and return the squared L2 error.
pub(crate) fn print_quality_report(true_model: &Array3<f64>, reconstructed: &Array3<f64>) -> f64 {
    let count = true_model.len() as f64;
    let l2 = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&truth, estimate)| (truth - estimate).powi(2))
        .sum::<f64>();
    let rmse = (l2 / count).sqrt();
    let mean_truth = true_model.iter().sum::<f64>() / count;
    let mean_estimate = reconstructed.iter().sum::<f64>() / count;
    let covariance = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&truth, estimate)| (truth - mean_truth) * (estimate - mean_estimate))
        .sum::<f64>();
    let truth_variance = true_model
        .iter()
        .map(|&value| (value - mean_truth).powi(2))
        .sum::<f64>();
    let estimate_variance = reconstructed
        .iter()
        .map(|&value| (value - mean_estimate).powi(2))
        .sum::<f64>();
    let denominator = (truth_variance * estimate_variance).sqrt();
    let maximum_error = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&truth, estimate)| (truth - estimate).abs())
        .fold(0.0_f64, f64::max);
    let within_100 = true_model
        .iter()
        .zip(reconstructed.iter())
        .filter(|(&truth, estimate)| (truth - *estimate).abs() <= 100.0)
        .count() as f64
        / count
        * 100.0;

    println!("  RMSE            : {rmse:8.1} m/s");
    if denominator > f64::EPSILON {
        println!("  Pearson r       : {:8.4}", covariance / denominator);
    } else {
        println!("  Pearson r       :      N/A  (uniform model)");
    }
    println!("  Max |error|     : {maximum_error:8.1} m/s");
    println!("  Voxels ±100 m/s : {within_100:7.1} %");
    l2
}

/// Print error metrics for a preselected set of truth/estimate pairs.
pub(crate) fn print_quality_pairs(pairs: &[(f64, f64)]) {
    let count = pairs.len() as f64;
    if count < 2.0 {
        println!("  (no free voxels)");
        return;
    }
    let l2 = pairs
        .iter()
        .map(|&(truth, estimate)| (truth - estimate).powi(2))
        .sum::<f64>();
    let rmse = (l2 / count).sqrt();
    let mean_truth = pairs.iter().map(|&(truth, _)| truth).sum::<f64>() / count;
    let mean_estimate = pairs.iter().map(|&(_, estimate)| estimate).sum::<f64>() / count;
    let covariance = pairs
        .iter()
        .map(|&(truth, estimate)| (truth - mean_truth) * (estimate - mean_estimate))
        .sum::<f64>();
    let truth_variance = pairs
        .iter()
        .map(|&(truth, _)| (truth - mean_truth).powi(2))
        .sum::<f64>();
    let estimate_variance = pairs
        .iter()
        .map(|&(_, estimate)| (estimate - mean_estimate).powi(2))
        .sum::<f64>();
    let maximum_error = pairs
        .iter()
        .map(|&(truth, estimate)| (truth - estimate).abs())
        .fold(0.0_f64, f64::max);
    let within_10 = pairs
        .iter()
        .filter(|&&(truth, estimate)| (truth - estimate).abs() <= 10.0)
        .count() as f64
        / count
        * 100.0;

    println!("  RMSE            : {rmse:8.2} m/s");
    let denominator = (truth_variance * estimate_variance).sqrt();
    if denominator > f64::EPSILON {
        println!("  Pearson r       : {:8.4}", covariance / denominator);
    }
    println!("  Max |error|     : {maximum_error:8.2} m/s");
    println!("  Voxels ±10 m/s  : {within_10:7.1} %");
}

#[cfg(test)]
mod tests {
    use super::print_quality_report;
    use leto::Array3;

    #[test]
    fn whole_model_report_returns_squared_l2() {
        let truth = Array3::from_elem((2, 1, 2), 3.0);
        let estimate = Array3::from_elem((2, 1, 2), 1.0);
        assert_eq!(print_quality_report(&truth, &estimate), 16.0);
    }
}
