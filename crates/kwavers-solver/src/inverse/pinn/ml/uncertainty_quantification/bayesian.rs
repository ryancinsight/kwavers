//! Bayesian PINN with uncertainty quantification via deep ensembles.

use super::types::{
    PinnPredictionWithUncertainty, PinnUncertaintyConfig, PinnUncertaintyMethod, UncertaintyStats,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use burn::tensor::backend::AutodiffBackend;
use ndarray::Array1;

use super::conformal::PinnConformalPredictor;

/// Bayesian PINN with uncertainty quantification.
#[derive(Debug)]
pub struct PinnBayesianPINN<B: AutodiffBackend> {
    /// Ensemble of models for uncertainty estimation.
    pub(super) ensemble: Vec<crate::inverse::pinn::ml::BurnPINN2DWave<B>>,
    /// Uncertainty configuration.
    pub(super) config: PinnUncertaintyConfig,
    /// Calibration data for conformal prediction.
    pub calibration_data: Option<Vec<(Vec<f32>, f32)>>,
    /// Conformal predictor for prediction intervals.
    pub conformal_predictor: Option<PinnConformalPredictor<B>>,
    /// Performance statistics.
    pub stats: UncertaintyStats,
}

impl<B: AutodiffBackend> PinnBayesianPINN<B> {
    /// Create a new Bayesian PINN.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        base_model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        config: PinnUncertaintyConfig,
    ) -> KwaversResult<Self> {
        let mut ensemble = Vec::new();

        for i in 0..config.ensemble_size {
            let _model_idx = i;
            let model = base_model.clone();
            ensemble.push(model);
        }

        Ok(Self {
            ensemble,
            config,
            calibration_data: None,
            conformal_predictor: None,
            stats: UncertaintyStats::default(),
        })
    }

    /// Calibrate uncertainty estimates using validation data.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calibrate(
        &mut self,
        calibration_inputs: &[Vec<f32>],
        calibration_targets: &[f32],
    ) -> KwaversResult<()> {
        let calibration_data = calibration_inputs
            .iter()
            .zip(calibration_targets.iter())
            .map(|(input, target)| (input.clone(), *target))
            .collect();

        self.calibration_data = Some(calibration_data);

        if self.config.conformal_alpha > 0.0 {
            let mut cp =
                PinnConformalPredictor::new(self.ensemble[0].clone(), self.config.conformal_alpha);
            cp.calibrate(calibration_inputs, calibration_targets)?;
            self.conformal_predictor = Some(cp);
        }

        Ok(())
    }

    /// Predict with uncertainty quantification.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn predict_with_uncertainty(
        &mut self,
        input: &[f32],
    ) -> KwaversResult<PinnPredictionWithUncertainty> {
        if self.config.mc_samples > 0 {
            return Err(KwaversError::InvalidInput(
                "MC dropout is not supported by the current Burn PINN architectures".into(),
            ));
        }
        self.ensemble_prediction(input)
    }

    /// Deep ensemble prediction.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn ensemble_prediction(
        &mut self,
        input: &[f32],
    ) -> KwaversResult<PinnPredictionWithUncertainty> {
        let mut predictions = Vec::new();

        for model in &self.ensemble {
            let prediction = self.model_prediction(model, input)?;
            predictions.push(prediction);
        }

        let mut stats =
            self.compute_uncertainty_stats(&predictions, PinnUncertaintyMethod::DeepEnsemble)?;

        if let Some(cp) = &self.conformal_predictor {
            let (lower, upper) = cp.predict_conformal(input)?;
            stats.confidence_interval = (vec![lower], vec![upper]);
            stats.method = PinnUncertaintyMethod::Hybrid;
        }

        Ok(stats)
    }

    /// Compute uncertainty statistics from predictions.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    fn compute_uncertainty_stats(
        &mut self,
        predictions: &[Vec<f32>],
        method: PinnUncertaintyMethod,
    ) -> KwaversResult<PinnPredictionWithUncertainty> {
        if predictions.is_empty() {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidOperation {
                    operation: "uncertainty_prediction".to_string(),
                    reason: "No predictions available".to_string(),
                },
            ));
        }

        let num_points = predictions[0].len();
        let num_samples = predictions.len();

        let mut means = vec![0.0; num_points];
        let mut variances = vec![0.0; num_points];

        for i in 0..num_points {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;

            for prediction in predictions {
                let val = prediction[i];
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / num_samples as f32;
            let variance = (sum_sq / num_samples as f32) - (mean * mean);

            means[i] = mean;
            variances[i] = variance.max(0.0);
        }

        let stds: Vec<f32> = variances.iter().map(|v| v.sqrt()).collect();
        let z_score = 1.96_f32; // 95% confidence

        let lower_bounds: Vec<f32> = means
            .iter()
            .zip(stds.iter())
            .map(|(m, s)| m - z_score * s)
            .collect();

        let upper_bounds: Vec<f32> = means
            .iter()
            .zip(stds.iter())
            .map(|(m, s)| m + z_score * s)
            .collect();

        let entropy = self.compute_predictive_entropy(&variances);
        let reliability = self.compute_reliability_score(&variances);

        let avg_uncertainty = variances.iter().sum::<f32>() / variances.len() as f32;
        self.stats.total_predictions += 1;
        self.stats.average_uncertainty = (self.stats.average_uncertainty + avg_uncertainty) / 2.0;

        Ok(PinnPredictionWithUncertainty {
            mean: means,
            std: stds,
            confidence_interval: (lower_bounds, upper_bounds),
            entropy,
            reliability,
            method,
        })
    }

    /// Get model prediction for ensemble member.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn model_prediction(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        input: &[f32],
    ) -> KwaversResult<Vec<f32>> {
        if input.len() != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }
        let x = Array1::from_elem((1,), input[0] as f64);
        let y = Array1::from_elem((1,), input[1] as f64);
        let t = Array1::from_elem((1,), input[2] as f64);
        let device = model.device();

        let output = model
            .predict(&x, &y, &t, &device)
            .map(|output| output.iter().map(|&v| v as f32).collect::<Vec<_>>())?;

        Ok(output)
    }

    /// Compute predictive entropy.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_predictive_entropy(&self, variances: &[f32]) -> f32 {
        let avg_variance = variances.iter().sum::<f32>() / variances.len() as f32;
        0.5 * (1.0 + (2.0 * std::f32::consts::PI * avg_variance).ln())
    }

    /// Compute reliability score.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_reliability_score(&self, variances: &[f32]) -> f32 {
        let avg_variance = variances.iter().sum::<f32>() / variances.len() as f32;
        1.0 / (1.0 + avg_variance / self.config.variance_threshold as f32)
    }

    /// Update calibration metrics.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn _update_calibration_metrics(
        &mut self,
        calibration_data: &[(Vec<f32>, f32)],
    ) -> KwaversResult<()> {
        if calibration_data.is_empty() {
            return Ok(());
        }

        let mut abs_errors = Vec::with_capacity(calibration_data.len());
        let mut covered = 0usize;

        for (input, target) in calibration_data {
            let prediction = self.ensemble_prediction(input)?;
            let mean = *prediction.mean.first().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })?;
            let std = *prediction.std.first().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty uncertainty".into())
            })?;

            let err = (mean - *target).abs();
            abs_errors.push(err);

            let z_score = 1.96_f32;
            let lower = mean - z_score * std;
            let upper = mean + z_score * std;
            if (*target >= lower) && (*target <= upper) {
                covered += 1;
            }
        }

        let calibration_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
        let coverage_probability = covered as f32 / calibration_data.len() as f32;

        self.stats.calibration_error = calibration_error;
        self.stats.coverage_probability = coverage_probability;
        self.stats.reliability_score =
            1.0 / (1.0 + calibration_error / self.config.variance_threshold as f32);

        Ok(())
    }

    /// Get uncertainty statistics.
    pub fn get_stats(&self) -> &UncertaintyStats {
        &self.stats
    }
}
