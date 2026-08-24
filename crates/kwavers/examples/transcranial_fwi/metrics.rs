//! Value-semantic reconstruction metrics for the transcranial demonstration.

use leto::Array3;

/// Print model error metrics and return the unnormalised squared L2 error.
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
        println!("  Pearson r       :      N/A  (uniform model — undefined)");
    }
    println!("  Max |error|     : {maximum_error:8.1} m/s");
    println!("  Voxels ±100 m/s : {within_100:7.1} %");
    l2
}

#[cfg(test)]
mod tests {
    use super::print_quality_report;
    use leto::Array3;

    #[test]
    fn quality_report_returns_value_semantic_squared_l2() {
        let truth = Array3::from_elem((2, 1, 2), 3.0);
        let estimate = Array3::from_elem((2, 1, 2), 1.0);
        assert_eq!(print_quality_report(&truth, &estimate), 16.0);
    }
}
