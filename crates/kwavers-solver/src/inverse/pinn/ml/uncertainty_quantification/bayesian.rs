//! Bayesian PINN with uncertainty quantification via deep ensembles.

use super::types::{
    PinnPredictionWithUncertainty, PinnUncertaintyConfig, PinnUncertaintyMethod, UncertaintyStats,
};
use super::{
    conformal::PinnConformalPredictor, precision::restore_model_precision,
    statistics::summarize_predictions,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;
use tyche_core::Moments;

// Standard-normal 97.5th percentile, rounded once to the model scalar.
const NORMAL_95_PERCENT_TWO_SIDED: f32 = 1.959_964;

/// Bayesian PINN with uncertainty quantification.
pub struct PinnBayesianPINN<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Ensemble of models for uncertainty estimation.
    pub(super) ensemble: Vec<crate::inverse::pinn::ml::PinnWave2D<B>>,
    /// Uncertainty configuration.
    pub(super) config: PinnUncertaintyConfig,
    /// Calibration data for conformal prediction.
    pub calibration_data: Option<Vec<(Vec<f32>, f32)>>,
    /// Conformal predictor for prediction intervals.
    pub conformal_predictor: Option<PinnConformalPredictor<B>>,
    /// Performance statistics.
    pub stats: UncertaintyStats,
    uncertainty_history: Moments<f32>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PinnBayesianPINN<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnBayesianPINN")
            .field("ensemble_size", &(self.ensemble.len()))
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnBayesianPINN<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new Bayesian PINN.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        base_model: &crate::inverse::pinn::ml::PinnWave2D<B>,
        config: PinnUncertaintyConfig,
    ) -> KwaversResult<Self> {
        if config.ensemble_size == 0 {
            return Err(KwaversError::InvalidInput(
                "PINN ensemble size must be non-zero".to_owned(),
            ));
        }
        if !config.conformal_alpha.is_finite()
            || config.conformal_alpha < 0.0
            || config.conformal_alpha >= 1.0
        {
            return Err(KwaversError::InvalidInput(format!(
                "PINN conformal alpha must be zero (disabled) or in (0, 1): {}",
                config.conformal_alpha
            )));
        }
        if !config.variance_threshold.is_finite() || config.variance_threshold <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PINN variance threshold must be positive and finite: {}",
                config.variance_threshold
            )));
        }
        let ensemble = vec![base_model.clone(); config.ensemble_size];

        Ok(Self {
            ensemble,
            config,
            calibration_data: None,
            conformal_predictor: None,
            stats: UncertaintyStats::default(),
            uncertainty_history: Moments::new(),
        })
    }

    /// Calibrate uncertainty estimates using validation data.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn calibrate(
        &mut self,
        calibration_inputs: &[Vec<f32>],
        calibration_targets: &[f32],
    ) -> KwaversResult<()> {
        if calibration_inputs.len() != calibration_targets.len() {
            return Err(KwaversError::InvalidInput(
                "Calibration inputs and targets must have same length".to_owned(),
            ));
        }
        if calibration_inputs.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Calibration data must be non-empty".to_owned(),
            ));
        }

        if self.config.conformal_alpha > 0.0 {
            let model = self.ensemble.first().ok_or_else(|| {
                KwaversError::InvalidInput("PINN ensemble must be non-empty".to_owned())
            })?;
            let mut cp = PinnConformalPredictor::new(model.clone(), self.config.conformal_alpha)?;
            cp.calibrate(calibration_inputs, calibration_targets)?;
            self.conformal_predictor = Some(cp);
        }

        self.calibration_data = Some(
            calibration_inputs
                .iter()
                .zip(calibration_targets)
                .map(|(input, &target)| (input.clone(), target))
                .collect(),
        );
        Ok(())
    }

    /// Predict with uncertainty quantification.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn predict_with_uncertainty(
        &mut self,
        input: &[f32],
    ) -> KwaversResult<PinnPredictionWithUncertainty> {
        if self.config.mc_samples > 0 {
            return Err(KwaversError::InvalidInput(
                "MC dropout is not supported by the current current PINN architectures".into(),
            ));
        }
        self.ensemble_prediction(input)
    }

    /// Deep ensemble prediction.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
    /// - Returns [`crate::KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    fn compute_uncertainty_stats(
        &mut self,
        predictions: &[Vec<f32>],
        method: PinnUncertaintyMethod,
    ) -> KwaversResult<PinnPredictionWithUncertainty> {
        let summary = summarize_predictions(predictions)?;
        let mean_variance = summary.mean_variance;
        let means = summary.means;
        let variances = summary.variances;

        let stds: Vec<f32> = variances.iter().map(|v| v.sqrt()).collect();

        let lower_bounds: Vec<f32> = means
            .iter()
            .zip(stds.iter())
            .map(|(mean, standard_deviation)| {
                mean - NORMAL_95_PERCENT_TWO_SIDED * standard_deviation
            })
            .collect();

        let upper_bounds: Vec<f32> = means
            .iter()
            .zip(stds.iter())
            .map(|(mean, standard_deviation)| {
                mean + NORMAL_95_PERCENT_TWO_SIDED * standard_deviation
            })
            .collect();

        let entropy = self.compute_predictive_entropy(mean_variance);
        let reliability = self.compute_reliability_score(mean_variance);

        self.uncertainty_history.update(mean_variance);
        self.stats.total_predictions += 1;
        self.stats.average_uncertainty = self.uncertainty_history.mean().map_err(|error| {
            KwaversError::InvalidInput(format!("Uncertainty history is undefined: {error}"))
        })?;

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
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    fn model_prediction(
        &self,
        model: &crate::inverse::pinn::ml::PinnWave2D<B>,
        input: &[f32],
    ) -> KwaversResult<Vec<f32>> {
        let [x, y, t] = input else {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        };
        if !x.is_finite() || !y.is_finite() || !t.is_finite() {
            return Err(KwaversError::InvalidInput(
                "PINN coordinates must be finite".into(),
            ));
        }
        let x = Array1::from_elem([1], f64::from(*x));
        let y = Array1::from_elem([1], f64::from(*y));
        let t = Array1::from_elem([1], f64::from(*t));

        let output = model.predict(&x, &y, &t).map(|output| {
            output
                .iter()
                .copied()
                .map(restore_model_precision)
                .collect::<Vec<_>>()
        })?;

        Ok(output)
    }

    /// Compute predictive entropy.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_predictive_entropy(&self, mean_variance: f32) -> f32 {
        0.5 * (1.0 + (2.0 * std::f32::consts::PI * mean_variance).ln())
    }

    /// Compute reliability score.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_reliability_score(&self, mean_variance: f32) -> f32 {
        1.0 / (1.0 + mean_variance / self.config.variance_threshold)
    }

    /// Get uncertainty statistics.
    pub fn get_stats(&self) -> &UncertaintyStats {
        &self.stats
    }
}
