//! Speed-shift reconstruction image result types.

use ndarray::Array2;

use super::config::{ShiftPrior, ShiftSampling};

/// Reconstructed speed-of-sound shift image.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftImage {
    /// Estimated `delta c = c - c0` [m/s] on the input mask grid.
    pub sound_speed_shift_m_s: Array2<f64>,
    /// Objective value after each solver iteration, including the initial state.
    pub objective_history: Vec<f64>,
    /// Number of selected measurement rows used by the inverse solve.
    pub rows_used: usize,
    /// Number of supplied measurement rows before sampling.
    pub rows_available: usize,
    /// Number of active image pixels in the reconstruction support.
    pub active_voxels: usize,
    /// Model identifier for audit trails.
    pub model_family: &'static str,
    /// Measurement-row policy used for this image.
    pub sampling: ShiftSampling,
    /// Image prior used for this image.
    pub prior: ShiftPrior,
}

/// Borrowed speed-of-sound shift image output.
///
/// This view is returned by allocation-preserving reconstruction APIs that
/// write image values into caller-owned storage. The references remain valid
/// until the caller reuses the output image or reconstruction workspace.
#[derive(Clone, Copy, Debug)]
pub struct SoundSpeedShiftImageView<'a> {
    /// Estimated `delta c = c - c0` [m/s] on the input mask grid.
    pub sound_speed_shift_m_s: &'a Array2<f64>,
    /// Objective value after each solver iteration, including the initial state.
    pub objective_history: &'a [f64],
    /// Number of selected measurement rows used by the inverse solve.
    pub rows_used: usize,
    /// Number of supplied measurement rows before sampling.
    pub rows_available: usize,
    /// Number of active image pixels in the reconstruction support.
    pub active_voxels: usize,
    /// Model identifier for audit trails.
    pub model_family: &'static str,
    /// Measurement-row policy used for this image.
    pub sampling: ShiftSampling,
    /// Image prior used for this image.
    pub prior: ShiftPrior,
}
