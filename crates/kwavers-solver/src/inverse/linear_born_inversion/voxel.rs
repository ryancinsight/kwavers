//! Per-voxel state for linear Born inversion.
//!
//! Anatomy-neutral container carrying everything the generic sensitivity
//! matrix + PCG kernels need per active voxel:
//!
//! - Grid indices `(ix, iy, iz)` for active-set lookup and edge-preserving
//!   neighbour traversal.
//! - Cartesian positions `(x_m, y_m, z_m)` for Born sensitivity evaluation
//!   (path lengths to transducer elements).
//! - `target_contrast`: scalar slowness contrast used by the synthetic-data
//!   path. Clinical adapters compute this from their medium model
//!   (CT-derived sound speed, MRI-derived sound speed, etc.).
//! - `attenuation_np_per_m_mhz`: scalar frequency-linear attenuation used by
//!   the optional attenuation-aware sensitivity row weighting. Same clinical
//!   adapter origin.
//!
//! The struct is value-only data; no methods, no medium pointer, no anatomy
//! constants. Generic kernels iterate it as `&[VolumeVoxel]`.

/// Per-voxel inversion state for the generic linear Born + PCG kernels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeVoxel {
    /// X grid index.
    pub ix: usize,
    /// Y grid index.
    pub iy: usize,
    /// Z grid index.
    pub iz: usize,
    /// Cartesian X position \[m\].
    pub x_m: f64,
    /// Cartesian Y position \[m\].
    pub y_m: f64,
    /// Cartesian Z position \[m\].
    pub z_m: f64,
    /// Slowness contrast (clinical-adapter supplied).
    pub target_contrast: f64,
    /// Frequency-linear attenuation in Np / (m·MHz) (clinical-adapter supplied).
    pub attenuation_np_per_m_mhz: f64,
}
