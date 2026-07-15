//! Monte Carlo inference: quantify_uncertainty and compute_prediction_statistics.

use super::{MlBayesianPINN, MlPredictionWithUncertainty};
#[cfg(feature = "pinn")]
use crate::ml::uncertainty::PinnUncertaintyPredictor;
use kwavers_core::error::KwaversResult;
use leto::Array2;
use std::collections::HashMap;

impl MlBayesianPINN {
    /// Quantify uncertainty for PINN predictions
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(feature = "pinn")]
    pub fn quantify_uncertainty<P: PinnUncertaintyPredictor + ?Sized>(
        &self,
        predictor: &P,
        inputs: &Array2<f32>,
    ) -> KwaversResult<MlPredictionWithUncertainty> {
        let mut predictions = Vec::new();

        for _ in 0..self._config.num_samples {
            predictions.push(predictor.predict_inputs(inputs)?);
        }

        self.compute_prediction_statistics(&predictions)
    }

    /// Compute prediction statistics from Monte Carlo samples.
    ///
    /// Returns mean, variance (Bessel-corrected std dev), 68% and 95% confidence
    /// intervals, and a reliability score ∈ (0, 1].
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if `predictions` is empty.
    ///
    pub(super) fn compute_prediction_statistics(
        &self,
        predictions: &[Array2<f32>],
    ) -> KwaversResult<MlPredictionWithUncertainty> {
        if predictions.is_empty() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "No predictions available for statistics".to_owned(),
            ));
        }

        let shape = predictions[0].shape();
        let mut mean_prediction = Array2::zeros(shape);
        let mut variance: Array2<f32> = Array2::zeros(shape);

        for prediction in predictions {
            mean_prediction = &mean_prediction + prediction;
        }
        mean_prediction = &mean_prediction / predictions.len() as f32;

        for prediction in predictions {
            let diff = prediction - &mean_prediction;
            variance = &variance + &(&diff * &diff);
        }
        variance = &variance / (predictions.len() - 1) as f32;

        let mut uncertainty = Array2::zeros(shape);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                uncertainty[[i, j]] = variance[[i, j]].sqrt();
            }
        }

        let mut confidence_intervals = HashMap::new();

        let z_score_95 = 1.96;
        let ci_95_lower = &mean_prediction - &(&uncertainty * z_score_95);
        let ci_95_upper = &mean_prediction + &(&uncertainty * z_score_95);
        confidence_intervals.insert("95%".to_owned(), (ci_95_lower, ci_95_upper));

        let ci_68_lower = &mean_prediction - &uncertainty;
        let ci_68_upper = &mean_prediction + &uncertainty;
        confidence_intervals.insert("68%".to_owned(), (ci_68_lower, ci_68_upper));

        let mean_uncertainty = uncertainty.iter().sum::<f32>() / uncertainty.len() as f32;
        let mean_prediction_magnitude = mean_prediction.iter().map(|x: &f32| x.abs()).sum::<f32>()
            / mean_prediction.len() as f32;

        let reliability_score = if mean_prediction_magnitude > 0.0 {
            1.0 / (1.0 + mean_uncertainty / mean_prediction_magnitude)
        } else {
            0.5
        };

        Ok(MlPredictionWithUncertainty {
            mean_prediction,
            uncertainty,
            confidence_intervals,
            reliability_score: reliability_score.into(),
        })
    }
}
