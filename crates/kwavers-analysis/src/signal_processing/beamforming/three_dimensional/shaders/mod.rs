//! WGSL compute shaders for 3D beamforming (GPU path).
//!
//! Co-located with the `three_dimensional` beamforming processor that consumes
//! them. GPU concretions live in the analysis layer (their only consumer), not
//! in the domain model layer.

/// Static delay-and-sum (plane-wave transmit). Entry: `delay_and_sum_main`.
/// GPU port of [`super::cpu::das::delay_and_sum_cpu`], differentially tested.
pub const BEAMFORMING_3D_SHADER: &str = include_str!("beamforming_3d.wgsl");

/// Dynamic (per-depth-zone) transmit-focus DAS. Entry: `dynamic_focus_main`.
pub const DYNAMIC_FOCUS_3D_SHADER: &str = include_str!("dynamic_focus_3d.wgsl");
