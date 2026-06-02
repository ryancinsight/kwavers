//! Repeated-frame reconstruction through a cached operator.

use kwavers_core::error::KwaversResult;

use super::super::reconstruct_from_operator;
use super::super::types::{SoundSpeedShiftImage, SoundSpeedShiftWorkspace};
use super::types::SoundSpeedShiftPlan;
use super::validation::validate_frame_time_shifts;

impl SoundSpeedShiftPlan {
    /// Reconstruct one frame using a temporary workspace.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when `time_shifts_s` does
    /// not match the fixed acquisition row contract.
    pub fn reconstruct(&self, time_shifts_s: &[f64]) -> KwaversResult<SoundSpeedShiftImage> {
        let mut workspace = SoundSpeedShiftWorkspace::new();
        self.reconstruct_with_workspace(time_shifts_s, &mut workspace)
    }

    /// Reconstruct one frame using caller-owned solver scratch buffers.
    ///
    /// `time_shifts_s` is indexed by the original acquisition rows, before any
    /// sparse sampling policy is applied.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when `time_shifts_s` does
    /// not match the fixed acquisition row contract.
    pub fn reconstruct_with_workspace(
        &self,
        time_shifts_s: &[f64],
        workspace: &mut SoundSpeedShiftWorkspace,
    ) -> KwaversResult<SoundSpeedShiftImage> {
        validate_frame_time_shifts(time_shifts_s, self.samples.len())?;
        let data = self
            .operator
            .rhs_from_time_shift_values(time_shifts_s, self.config.reference_sound_speed_m_s);
        Ok(reconstruct_from_operator(
            &self.operator,
            &data,
            self.samples.len(),
            self.config,
            workspace,
        ))
    }
}
