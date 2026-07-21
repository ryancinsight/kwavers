//! Tyche conformal prediction for PINN uncertainty quantification.

use super::precision::restore_model_precision;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;
use tyche_core::{ConformalCalibrator, ConformalError};

/// Conformal prediction for uncertainty quantification.
pub struct PinnConformalPredictor<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    model: crate::inverse::pinn::ml::PinnWave2D<B>,
    calibration_scores: Vec<f32>,
    calibrator: ConformalCalibrator<f32>,
    quantile: Option<f32>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PinnConformalPredictor<B>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnConformalPredictor")
            .field("calibration_scores_len", &self.calibration_scores.len())
            .field("miscoverage", &self.calibrator.miscoverage())
            .field("quantile", &self.quantile)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnConformalPredictor<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new conformal predictor with validated `f32` miscoverage.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] unless `0 < miscoverage < 1`.
    pub fn new(
        model: crate::inverse::pinn::ml::PinnWave2D<B>,
        miscoverage: f32,
    ) -> KwaversResult<Self> {
        let calibrator = ConformalCalibrator::new(miscoverage)
            .map_err(|error| conformal_error("invalid PINN miscoverage", error))?;
        Ok(Self {
            model,
            calibration_scores: Vec::new(),
            calibrator,
            quantile: None,
        })
    }

    /// Calibrate using prediction inputs and scalar targets.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] for mismatched or empty
    /// calibration data, invalid inputs, or non-finite scores. Propagates model
    /// prediction failures.
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

        let mut scores = Vec::with_capacity(calibration_inputs.len());
        for (input, &target) in calibration_inputs.iter().zip(calibration_targets) {
            scores.push(self.compute_nonconformity_score(input, target)?);
        }
        let quantile = self
            .calibrator
            .calibrate_in_place(&mut scores)
            .map_err(|error| conformal_error("invalid PINN calibration scores", error))?;

        self.calibration_scores = scores;
        self.quantile = Some(quantile);
        Ok(())
    }

    /// Predict with a symmetric conformal interval.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] before calibration or for an
    /// invalid coordinate input. Propagates model prediction failures.
    pub fn predict_conformal(&self, input: &[f32]) -> KwaversResult<(f32, f32)> {
        let quantile = self.quantile.ok_or_else(|| {
            KwaversError::InvalidInput(
                "PinnConformalPredictor must be calibrated before prediction".into(),
            )
        })?;
        let center = self.predict_scalar(input)?;
        let interval = self.calibrator.interval(center, quantile);
        Ok((interval.lower(), interval.upper()))
    }

    /// Validated miscoverage.
    #[must_use]
    pub const fn miscoverage(&self) -> f32 {
        self.calibrator.miscoverage()
    }

    /// Borrow sorted calibration scores.
    #[must_use]
    pub fn calibration_scores(&self) -> &[f32] {
        &self.calibration_scores
    }

    /// Calibrated radius, if calibration has completed.
    #[must_use]
    pub const fn calibrated_radius(&self) -> Option<f32> {
        self.quantile
    }

    fn compute_nonconformity_score(&self, input: &[f32], target: f32) -> KwaversResult<f32> {
        if !target.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Calibration target must be finite: {target}"
            )));
        }
        let prediction = self.predict_scalar(input)?;
        let score = (prediction - target).abs();
        if !score.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "PINN nonconformity score is non-finite: {score}"
            )));
        }
        Ok(score)
    }

    fn predict_scalar(&self, input: &[f32]) -> KwaversResult<f32> {
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
        let prediction = self.model.predict(&x, &y, &t)?;
        let value =
            restore_model_precision(*prediction.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })?);
        if !value.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Model returned non-finite prediction: {value}"
            )));
        }
        Ok(value)
    }
}

fn conformal_error<T: std::fmt::Debug>(context: &str, error: ConformalError<T>) -> KwaversError {
    KwaversError::InvalidInput(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests;
