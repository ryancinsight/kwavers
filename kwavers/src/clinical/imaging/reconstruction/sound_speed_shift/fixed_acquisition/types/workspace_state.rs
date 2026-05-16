//! Fixed-acquisition workspace state.

use super::super::super::types::SoundSpeedShiftWorkspace;

/// Reusable buffers for fixed-acquisition single-frame reconstruction.
#[derive(Clone, Debug, Default)]
pub struct SoundSpeedShiftPlanWorkspace {
    pub(in super::super) sampled_rhs: Vec<f64>,
    pub(in super::super) solver: SoundSpeedShiftWorkspace,
}
