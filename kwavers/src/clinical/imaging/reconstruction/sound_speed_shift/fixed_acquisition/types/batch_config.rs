//! Fixed-acquisition batch policy types.

/// Objective-history retention policy for batch reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoundSpeedShiftObjectiveHistoryPolicy {
    /// Store only initial/final objective values and iteration count.
    Compact,
    /// Store the full objective history for every frame.
    Full,
}

/// Per-frame image retention policy for batch reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoundSpeedShiftBatchImageRetention {
    /// Retain every reconstructed image in the batch output.
    Retain,
    /// Discard reconstructed images and retain summaries/objective evidence.
    Discard,
}

/// Batch reconstruction output policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundSpeedShiftBatchConfig {
    /// Per-frame objective-history retention policy.
    pub objective_history: SoundSpeedShiftObjectiveHistoryPolicy,
    /// Per-frame reconstructed-image retention policy.
    pub image_retention: SoundSpeedShiftBatchImageRetention,
}

impl Default for SoundSpeedShiftBatchConfig {
    fn default() -> Self {
        Self {
            objective_history: SoundSpeedShiftObjectiveHistoryPolicy::Compact,
            image_retention: SoundSpeedShiftBatchImageRetention::Retain,
        }
    }
}

impl SoundSpeedShiftBatchConfig {
    /// Select per-frame objective-history retention.
    #[must_use]
    pub fn with_objective_history(
        mut self,
        objective_history: SoundSpeedShiftObjectiveHistoryPolicy,
    ) -> Self {
        self.objective_history = objective_history;
        self
    }

    /// Select per-frame reconstructed-image retention.
    #[must_use]
    pub fn with_image_retention(
        mut self,
        image_retention: SoundSpeedShiftBatchImageRetention,
    ) -> Self {
        self.image_retention = image_retention;
        self
    }
}
