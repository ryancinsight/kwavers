//! GPU Minimum Variance Distortionless Response (MVDR) beamforming kernel.
//!
//! ## Algorithm — Capon 1969
//!
//! The GPU path implements the same spatially-smoothed covariance MVDR
//! algorithm as [`super::cpu::mvdr`]: for each voxel, build R̂ from
//! overlapping sub-aperture covariance estimates, apply relative diagonal
//! loading δ, Cholesky-factor R_δ, solve R_δ u = 1, and return
//! |P · uᵀ x̄₀| with P = 1/(1ᵀ u).
//!
//! All inner-loop computation runs in `mvdr_3d.wgsl` with one GPU invocation
//! per voxel (workgroup_size = 1×1×1).  The covariance accumulation is
//! O(Q · L² · N) and the Cholesky solve is O(L³) per voxel; L ≤ 32 is
//! enforced at the Rust call site.
//!
//! ## References
//! - Capon J. (1969) IEEE Proc. 57(8) 1408–1418.
//! - Synnevåg J.F., Austeng A., Holm S. (2007) IEEE TUFFC 54(8) 1606–1613.
//! - Shan T.J., Kailath T. (1985) IEEE TASSP 33(3) 527–536.

mod dispatch;
mod params;

#[cfg(feature = "gpu")]
use crate::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;
#[cfg(feature = "gpu")]
use kwavers_core::error::KwaversResult;
#[cfg(feature = "gpu")]
use ndarray::{Array3, Array4};

/// GPU MVDR beamforming dispatcher.
///
/// Holds borrowed references to GPU device/queue/pipeline/layout owned by
/// [`super::processor::BeamformingProcessor3D`].  Only available when the
/// `gpu` feature is enabled.
#[cfg(feature = "gpu")]
pub(super) struct MvdrGPU<'a> {
    pub(super) config: &'a BeamformingConfig3D,
    pub(super) device: &'a wgpu::Device,
    pub(super) queue: &'a wgpu::Queue,
    pub(super) pipeline: &'a wgpu::ComputePipeline,
    pub(super) bind_group_layout: &'a wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl<'a> MvdrGPU<'a> {
    /// Create a GPU MVDR dispatcher from borrowed processor resources.
    pub(super) fn new(
        config: &'a BeamformingConfig3D,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        pipeline: &'a wgpu::ComputePipeline,
        bind_group_layout: &'a wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            config,
            device,
            queue,
            pipeline,
            bind_group_layout,
        }
    }

    /// Dispatch MVDR beamforming and return the reconstructed 3D volume.
    ///
    /// `diagonal_loading` is the relative loading factor δ (typical: 1/L to
    /// 100/L).  `subarray_size` = \[lx, ly, lz\] with lx·ly·lz ≤ 32.
    ///
    /// # Errors
    /// - [`KwaversError::System`] if GPU buffer or pipeline operations fail.
    pub(super) fn process(
        &self,
        rf_data: &Array4<f32>,
        diagonal_loading: f32,
        subarray_size: [usize; 3],
    ) -> KwaversResult<Array3<f32>> {
        dispatch::mvdr_dispatch(self, rf_data, diagonal_loading, subarray_size)
    }
}
