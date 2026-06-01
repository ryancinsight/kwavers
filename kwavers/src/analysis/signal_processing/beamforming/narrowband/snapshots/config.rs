//! BasebandSnapshotConfig — legacy analytic-signal snapshot parameters

use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::core::error::{KwaversError, KwaversResult};

/// Snapshot extraction configuration for **legacy** analytic-signal complex baseband.
///
/// Prefer windowed snapshots via `SnapshotSelection` unless you have a specific reason to keep a
/// sample-decimation snapshot model.
#[derive(Debug, Clone)]
pub struct BasebandSnapshotConfig {
    /// Sampling frequency (Hz). Must be finite and > 0.
    pub sampling_frequency_hz: f64,
    /// Center frequency (Hz) used for complex downconversion. Must be finite and > 0.
    pub center_frequency_hz: f64,
    /// Snapshot stride in samples (>= 1). Each snapshot is taken from one time sample of the
    /// complex baseband after downconversion. Using a stride > 1 reduces snapshot count.
    pub snapshot_step_samples: usize,
}

impl BasebandSnapshotConfig {
    /// Validate invariants (mathematically necessary).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.sampling_frequency_hz.is_finite() || self.sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BasebandSnapshotConfig: sampling_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !self.center_frequency_hz.is_finite() || self.center_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BasebandSnapshotConfig: center_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if self.snapshot_step_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "BasebandSnapshotConfig: snapshot_step_samples must be >= 1".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for BasebandSnapshotConfig {
    fn default() -> Self {
        Self {
            sampling_frequency_hz: MHZ_TO_HZ,
            center_frequency_hz: 200_000.0,
            snapshot_step_samples: 1,
        }
    }
}
