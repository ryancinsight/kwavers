//! Provider contract for 3-D beamforming GPU execution.

use super::config::{Beamforming3dApodizationWindow, BeamformingConfig3D};
use kwavers_core::error::KwaversResult;
use kwavers_solver::backend::traits::GpuProvider;
use leto::{
    Array3,
    Array4,
};

/// 3-D beamforming GPU provider contract.
///
/// A provider owns concrete device acquisition, shader/kernel compilation, and
/// dispatch. CUDA enters this seam only when real CUDA delay-and-sum kernels
/// exist; it should not inherit WGPU handles or placeholder WGSL behavior.
pub trait BeamformingGpuProvider: std::fmt::Debug {
    /// Provider identity surfaced at the solver/backend boundary.
    fn provider_kind(&self) -> GpuProvider;

    /// Process a delay-and-sum reconstruction.
    ///
    /// # Errors
    ///
    /// Propagates provider transfer, dispatch, or readback failures.
    fn process_delay_and_sum(
        &self,
        config: &BeamformingConfig3D,
        rf_data: &Array4<f32>,
        dynamic_focusing: bool,
        apodization_window: &Beamforming3dApodizationWindow,
        apodization_weights: &Array3<f32>,
        sub_volume_size: Option<(usize, usize, usize)>,
    ) -> KwaversResult<Array3<f32>>;
}
