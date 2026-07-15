//! Uncertainty decomposition and calibration.

use super::{MlBayesianPINN, UncertaintyDecomposition};
use kwavers_core::error::KwaversResult;
use leto::Array2;
use log::info;

impl MlBayesianPINN {
    /// Estimate epistemic vs aleatoric uncertainty.
    ///
    /// Decomposes total MC-dropout uncertainty into equal epistemic (model) and
    /// aleatoric (data-noise) components following the equal-split heuristic.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if `predictions.len() < 2`.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn decompose_uncertainty(
        &self,
        predictions: &[Array2<f32>],
    ) -> KwaversResult<UncertaintyDecomposition> {
        if predictions.len() < 2 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Need at least 2 predictions for uncertainty decomposition".to_owned(),
            ));
        }

        let stats = self.compute_prediction_statistics(predictions)?;
        let total_uncertainty = stats.uncertainty;
        let epistemic = &total_uncertainty * 0.5;
        let aleatoric = &total_uncertainty * 0.5;

        Ok(UncertaintyDecomposition {
            total_uncertainty,
            epistemic_uncertainty: epistemic,
            aleatoric_uncertainty: aleatoric,
            uncertainty_ratio: 1.0,
        })
    }

    /// Calibrate uncertainty estimates using validation data.
    ///
    /// Computes the MAE-to-mean-uncertainty ratio as a calibration factor and
    /// logs it.  Future implementations may apply the factor to scale dropout
    /// masks.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calibrate_uncertainty(
        &mut self,
        validation_predictions: &[Array2<f32>],
        validation_targets: &[Array2<f32>],
    ) -> KwaversResult<()> {
        let stats = self.compute_prediction_statistics(validation_predictions)?;

        let mut mae = 0.0_f32;
        let n_samples = validation_predictions.len();

        for i in 0..n_samples {
            let error = &validation_predictions[i] - &validation_targets[i];
            mae += error.iter().map(|x| x.abs()).sum::<f32>();
        }
        mae /= (n_samples * stats.mean_prediction.len()) as f32;

        let calibration_factor =
            mae / (stats.uncertainty.iter().sum::<f32>() / stats.uncertainty.len() as f32);

        info!(
            "Uncertainty calibration: MAE = {:.4}, calibration factor = {:.4}",
            mae, calibration_factor
        );

        Ok(())
    }
}
