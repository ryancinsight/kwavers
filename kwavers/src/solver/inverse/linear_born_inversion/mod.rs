//! Generic linear single-scatter Born inversion primitives.
//!
//! Hosts the transducer-geometry abstraction and (after T13b lands) the
//! generic linear-PCG Born inversion core that is currently fused with
//! anatomy/transducer-specific code inside
//! `clinical::imaging::reconstruction::transcranial_ust`. The clinical
//! adapter retains the CT-resampling, head-volume, and bowl-array specifics
//! while delegating to this solver module for the generic numerical kernel.
//!
//! # Distinction from `fwi`
//!
//! - `solver::inverse::fwi::{frequency_domain, time_domain}` are nonlinear
//!   FWI: iterative inversion over the *forward operator itself* (Born
//!   series, FDTD/PSTD time stepping, full-waveform misfit).
//! - `solver::inverse::linear_born_inversion` is *linear* PCG inversion on a
//!   fixed single-scatter Born sensitivity matrix `Ax = d`. No iteration on
//!   the forward operator. Cheaper, regime-limited.
//!
//! Both inhabit `solver::inverse` because they are reconstruction methods;
//! the distinction lives in the module names, not in a misleading "Fwi"
//! identifier (closed in T11).

pub mod config;
pub(crate) mod dense;
pub mod enhancement;
pub mod geometry;
pub(crate) mod schedule;
pub mod voxel;

pub use config::LinearBornInversionConfig;
pub use enhancement::high_pass_enhance_volume;
pub use geometry::{ElementPosition, TransducerGeometry};
pub use voxel::VolumeVoxel;
