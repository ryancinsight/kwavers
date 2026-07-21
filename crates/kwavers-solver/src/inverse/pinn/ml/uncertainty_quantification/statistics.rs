//! Stable ensemble summaries for PINN uncertainty.

use kwavers_core::error::{KwaversError, KwaversResult, SystemError};
use tyche_core::{Moments, PopulationVariance};

#[derive(Debug)]
pub(super) struct EnsembleSummary {
    pub(super) means: Vec<f32>,
    pub(super) variances: Vec<f32>,
    pub(super) mean_variance: f32,
}

pub(super) fn summarize_predictions(predictions: &[Vec<f32>]) -> KwaversResult<EnsembleSummary> {
    let Some(first) = predictions.first() else {
        return Err(KwaversError::System(SystemError::InvalidOperation {
            operation: "uncertainty_prediction".to_owned(),
            reason: "No predictions available".to_owned(),
        }));
    };
    if first.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Ensemble predictions must contain at least one point".to_owned(),
        ));
    }
    let point_count = first.len();
    for (sample, prediction) in predictions.iter().enumerate() {
        if prediction.len() != point_count {
            return Err(KwaversError::InvalidInput(format!(
                "Ensemble sample {sample} has {} points; expected {point_count}",
                prediction.len()
            )));
        }
    }

    let mut means = Vec::with_capacity(point_count);
    let mut variances = Vec::with_capacity(point_count);
    let mut variance_moments = Moments::new();
    for point in 0..point_count {
        let mut moments = Moments::new();
        for (sample, prediction) in predictions.iter().enumerate() {
            let value = prediction[point];
            if !value.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "Ensemble sample {sample}, point {point} is non-finite: {value}"
                )));
            }
            moments.update(value);
        }
        let mean = moments.mean().map_err(|error| {
            KwaversError::InvalidInput(format!("Ensemble mean is undefined: {error}"))
        })?;
        let variance = moments.variance::<PopulationVariance>().map_err(|error| {
            KwaversError::InvalidInput(format!("Ensemble variance is undefined: {error}"))
        })?;
        means.push(mean);
        variances.push(variance);
        variance_moments.update(variance);
    }
    let mean_variance = variance_moments.mean().map_err(|error| {
        KwaversError::InvalidInput(format!("Mean ensemble variance is undefined: {error}"))
    })?;

    Ok(EnsembleSummary {
        means,
        variances,
        mean_variance,
    })
}

#[cfg(test)]
mod tests {
    use super::summarize_predictions;

    #[test]
    fn welford_preserves_variance_under_large_offset() {
        let summary = summarize_predictions(&[vec![10_000.0], vec![10_002.0]]).unwrap();
        assert_eq!(summary.means[0].to_bits(), 10_001.0_f32.to_bits());
        assert_eq!(summary.variances[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(summary.mean_variance.to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn shape_and_finiteness_failures_name_the_invalid_sample() {
        let shape_error = summarize_predictions(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert!(
            format!("{shape_error:?}").contains("sample 1"),
            "shape error must identify the mismatched sample"
        );

        let finite_error = summarize_predictions(&[vec![1.0], vec![f32::NAN]]).unwrap_err();
        assert!(
            format!("{finite_error:?}").contains("sample 1, point 0"),
            "finiteness error must identify sample and point"
        );
    }
}
