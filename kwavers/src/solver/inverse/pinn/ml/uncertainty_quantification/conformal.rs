//! Conformal prediction for PINN uncertainty quantification.

use crate::core::error::{KwaversError, KwaversResult};
use burn::tensor::backend::Backend;
use ndarray::Array1;

/// Conformal prediction for uncertainty quantification.
#[derive(Debug)]
pub struct ConformalPredictor<B: Backend> {
    /// Base PINN model.
    pub(super) model: crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
    /// Conformal scores from calibration.
    pub calibration_scores: Vec<f32>,
    /// Conformal alpha (significance level).
    pub alpha: f64,
    /// Computed quantile from calibration.
    pub quantile: Option<f32>,
}

impl<B: Backend> ConformalPredictor<B> {
    /// Create a new conformal predictor.
    pub fn new(model: crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>, alpha: f64) -> Self {
        let alpha = alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        Self {
            model,
            calibration_scores: Vec::new(),
            alpha,
            quantile: None,
        }
    }

    /// Calibrate using calibration data.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn calibrate(
        &mut self,
        calibration_inputs: &[Vec<f32>],
        calibration_targets: &[f32],
    ) -> KwaversResult<()> {
        if calibration_inputs.len() != calibration_targets.len() {
            return Err(KwaversError::InvalidInput(
                "Calibration inputs and targets must have same length".into(),
            ));
        }

        let mut scores: Vec<f32> = Vec::with_capacity(calibration_inputs.len());

        for (input, target) in calibration_inputs.iter().zip(calibration_targets.iter()) {
            let score = self.compute_nonconformity_score(input, *target)?;
            scores.push(score);
        }

        if scores.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Calibration dataset must be non-empty".into(),
            ));
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = scores.len();
        let k = (((n as f64 + 1.0) * (1.0 - self.alpha)).ceil() as usize).clamp(1, n);
        let q_hat = scores[k - 1];

        self.calibration_scores = scores;
        self.quantile = Some(q_hat);

        Ok(())
    }

    /// Predict with conformal uncertainty intervals.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn predict_conformal(&self, input: &[f32]) -> KwaversResult<(f32, f32)> {
        let q_hat = self.quantile.ok_or_else(|| {
            KwaversError::InvalidInput(
                "ConformalPredictor must be calibrated before prediction".into(),
            )
        })?;

        if input.len() != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }

        let device = self.model.device();
        let x = Array1::from_elem((1,), input[0] as f64);
        let y = Array1::from_elem((1,), input[1] as f64);
        let t = Array1::from_elem((1,), input[2] as f64);

        let pred = self.model.predict(&x, &y, &t, &device)?;
        let center =
            *pred.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })? as f32;

        Ok((center - q_hat, center + q_hat))
    }

    /// Compute nonconformity score.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn compute_nonconformity_score(&self, input: &[f32], target: f32) -> KwaversResult<f32> {
        if input.len() != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }

        let device = self.model.device();
        let x = Array1::from_elem((1,), input[0] as f64);
        let y = Array1::from_elem((1,), input[1] as f64);
        let t = Array1::from_elem((1,), input[2] as f64);

        let pred = self.model.predict(&x, &y, &t, &device)?;
        let y_hat =
            *pred.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })? as f32;

        Ok((y_hat - target).abs())
    }
}
