//! Construction and metadata access for fixed acquisition plans.

use ndarray::Array2;

use crate::core::error::KwaversResult;

use super::super::curved_array::CurvedArrayShiftScan;
use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{SoundSpeedShiftConfig, SoundSpeedShiftSample};
use super::types::SoundSpeedShiftPlan;

impl SoundSpeedShiftPlan {
    /// Build a plan from fixed transmit/receive geometry.
    ///
    /// The `time_shift_s` values already present in `samples` are ignored
    /// during reconstruction; callers provide each frame through
    /// [`SoundSpeedShiftPlan::reconstruct_with_workspace`].
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when geometry, sampling,
    /// propagation, sensitivity, or active-mask inputs violate the operator
    /// contract.
    pub fn new(
        samples: Vec<SoundSpeedShiftSample>,
        active_mask: &Array2<bool>,
        config: SoundSpeedShiftConfig,
    ) -> KwaversResult<Self> {
        let operator = SoundSpeedShiftOperator::new(&samples, active_mask, config)?;
        Ok(Self {
            samples,
            operator,
            config,
            shape: active_mask.dim(),
        })
    }

    /// Build a plan from a deterministic curved-array scan.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the scan, active mask,
    /// or reconstruction configuration violates its contract.
    pub fn from_curved_array_scan(
        scan: &CurvedArrayShiftScan,
        active_mask: &Array2<bool>,
        config: SoundSpeedShiftConfig,
    ) -> KwaversResult<Self> {
        Self::new(scan.samples()?, active_mask, config)
    }

    /// Fixed acquisition samples in original row order.
    #[must_use]
    pub fn samples(&self) -> &[SoundSpeedShiftSample] {
        &self.samples
    }

    /// Configuration used to assemble the cached operator.
    #[must_use]
    pub fn config(&self) -> SoundSpeedShiftConfig {
        self.config
    }

    /// Active image grid shape fixed by the plan.
    #[must_use]
    pub fn image_shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Number of frame shifts expected in original acquisition row order.
    #[must_use]
    pub fn rows_available(&self) -> usize {
        self.samples.len()
    }

    /// Number of selected rows in the cached operator after sampling.
    #[must_use]
    pub fn rows_used(&self) -> usize {
        self.operator.rows()
    }

    /// Number of active reconstruction pixels.
    #[must_use]
    pub fn active_voxels(&self) -> usize {
        self.operator.cols()
    }

    /// Number of nonzero geometric or finite-frequency weights retained.
    #[must_use]
    pub fn stored_weight_count(&self) -> usize {
        self.operator.stored_weight_count()
    }
}
