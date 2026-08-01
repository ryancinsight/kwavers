//! Speed-shift reconstruction image result types.

use aequitas::systems::si::quantities::Velocity;
use leto::Array2;

use super::config::{ShiftPrior, ShiftSampling};

/// Dense sound-speed shift field backed by Leto storage.
///
/// The provider array stores real base-unit values for solver interoperability.
/// Use [`Self::iter`] for velocity-typed values; use [`Self::storage`] only at
/// an explicit Leto/provider boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SoundSpeedShiftField {
    storage: Array2<f64>,
}

impl SoundSpeedShiftField {
    /// Wrap a Leto field whose values are measured in metres per second.
    ///
    /// This is the explicit provider-storage boundary. The array is retained
    /// without conversion so solver kernels can consume it directly.
    #[must_use]
    pub(crate) fn from_storage(storage: Array2<f64>) -> Self {
        Self { storage }
    }

    /// Borrow the underlying Leto storage for a provider or solver operation.
    #[must_use]
    pub(crate) fn storage(&self) -> &Array2<f64> {
        &self.storage
    }

    pub(crate) fn storage_mut(&mut self) -> &mut Array2<f64> {
        &mut self.storage
    }

    /// Return the field shape in row-major image order.
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        self.storage.shape()
    }

    /// Iterate over field values as Aequitas velocities.
    pub fn iter(&self) -> impl Iterator<Item = Velocity> + '_ {
        self.storage.iter().map(|value| Velocity::from_base(*value))
    }
}

/// Reconstructed speed-of-sound shift image.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftImage {
    /// Estimated `delta c = c - c0` on the input mask grid.
    pub sound_speed_shift: SoundSpeedShiftField,
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
    /// Estimated `delta c = c - c0` on the input mask grid.
    pub sound_speed_shift: &'a SoundSpeedShiftField,
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
