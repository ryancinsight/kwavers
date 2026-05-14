//! Forward prediction through a cached fixed-acquisition operator.

use ndarray::Array2;

use crate::core::error::{KwaversError, KwaversResult};

use super::types::SoundSpeedShiftPlan;

impl SoundSpeedShiftPlan {
    /// Predict selected-row travel-time shifts for a known speed-shift image.
    ///
    /// The output row order is the cached operator row order after
    /// [`super::super::types::ShiftSampling`] has been applied. Build the plan
    /// with dense sampling when every acquisition row is required.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the image shape does
    /// not match the plan's active-mask shape.
    pub fn predict_time_shifts(
        &self,
        sound_speed_shift_m_s: &Array2<f64>,
    ) -> KwaversResult<Vec<f64>> {
        if sound_speed_shift_m_s.dim() != self.shape {
            return Err(KwaversError::DimensionMismatch(format!(
                "speed-shift image shape {:?} does not match fixed plan shape {:?}",
                sound_speed_shift_m_s.dim(),
                self.shape
            )));
        }

        let model = self.operator.model_from_image(sound_speed_shift_m_s);
        let mut path_integrals = vec![0.0; self.operator.rows()];
        self.operator.matvec(&model, &mut path_integrals);
        let c0_sq = self.config.reference_sound_speed_m_s * self.config.reference_sound_speed_m_s;

        Ok(path_integrals
            .into_iter()
            .map(|value| -value / c0_sq)
            .collect())
    }
}
