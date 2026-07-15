//! WGSL compute shaders for 3D beamforming (GPU path).
//!
//! Co-located with the `kwavers-gpu` provider that consumes them. The analysis
//! crate owns the operation contract and CPU reference; concrete WGSL assets
//! live in this GPU leaf crate.

/// Static delay-and-sum (plane-wave transmit). Entry: `delay_and_sum_main`.
/// GPU port of the analysis-layer CPU delay-and-sum reference.
pub const BEAMFORMING_3D_SHADER: &str = include_str!("beamforming_3d.wgsl");

/// Dynamic (per-depth-zone) transmit-focus DAS. Entry: `dynamic_focus_main`.
pub const DYNAMIC_FOCUS_3D_SHADER: &str = include_str!("dynamic_focus_3d.wgsl");
