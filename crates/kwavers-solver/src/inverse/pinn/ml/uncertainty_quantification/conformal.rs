//! Conformal prediction for PINN uncertainty quantification.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

/// Conformal prediction for uncertainty quantification.
pub struct PinnConformalPredictor<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Base PINN model.
    pub(super) model: crate::inverse::pinn::ml::PinnWave2D<B>,
    /// Conformal scores from calibration.
    pub calibration_scores: Vec<f32>,
    /// Conformal alpha (significance level).
    pub alpha: f64,
    /// Computed quantile from calibration.
    pub quantile: Option<f32>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PinnConformalPredictor<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnConformalPredictor")
            .field("calibration_scores_len", &(self.calibration_scores.len()))
            .field("alpha", &self.alpha)
            .field("quantile", &self.quantile)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnConformalPredictor<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new conformal predictor.
    pub fn new(model: crate::inverse::pinn::ml::PinnWave2D<B>, alpha: f64) -> Self {
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
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn calibrate(
        &mut self,
        calibration_inputs: &[Vec<f32>],
        calibration_targets: &[f32],
    ) -> KwaversResult<()> {
        if (calibration_inputs.len()) != (calibration_targets.len()) {
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

        scores.sort_by(|a, b| a.total_cmp(b));
        let n = scores.len();
        let k = (((n as f64 + 1.0) * (1.0 - self.alpha)).ceil() as usize).clamp(1, n);
        let q_hat = scores[k - 1];

        self.calibration_scores = scores;
        self.quantile = Some(q_hat);

        Ok(())
    }

    /// Predict with conformal uncertainty intervals.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn predict_conformal(&self, input: &[f32]) -> KwaversResult<(f32, f32)> {
        let q_hat = self.quantile.ok_or_else(|| {
            KwaversError::InvalidInput(
                "PinnConformalPredictor must be calibrated before prediction".into(),
            )
        })?;

        if (input.len()) != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }

        let x = Array1::from_elem([1], input[0] as f64);
        let y = Array1::from_elem([1], input[1] as f64);
        let t = Array1::from_elem([1], input[2] as f64);

        let pred = self.model.predict(&x, &y, &t)?;
        let center =
            *pred.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })? as f32;

        Ok((center - q_hat, center + q_hat))
    }

    /// Compute nonconformity score.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    fn compute_nonconformity_score(&self, input: &[f32], target: f32) -> KwaversResult<f32> {
        if (input.len()) != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }

        let x = Array1::from_elem([1], input[0] as f64);
        let y = Array1::from_elem([1], input[1] as f64);
        let t = Array1::from_elem([1], input[2] as f64);

        let pred = self.model.predict(&x, &y, &t)?;
        let y_hat =
            *pred.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })? as f32;

        Ok((y_hat - target).abs())
    }
}
