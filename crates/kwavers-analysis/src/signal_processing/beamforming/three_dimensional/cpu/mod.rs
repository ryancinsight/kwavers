//! CPU beamforming kernels for 3D ultrasound.
//!
//! Provides Rayon-parallelised, analytically specified implementations of:
//! - Delay-and-Sum (DAS) — coherent receive summation with apodization.
//! - Minimum Variance Distortionless Response (MVDR / Capon) — adaptive
//!   covariance beamforming with spatial smoothing and diagonal loading.
//!
//! These are the authoritative CPU execution path, active whenever the `gpu`
//! feature flag is absent.  They also serve as the numerical reference baseline
//! for GPU kernel validation.
//!
//! # Sub-modules
//! - [`das`]: Delay-and-Sum with fractional-delay linear interpolation and
//!   full apodization support.
//! - [`mvdr`]: MVDR with spatially-smoothed covariance (Shan & Kailath 1985),
//!   relative diagonal loading, and Cholesky/LU solve via nalgebra.

pub mod das;
pub mod mvdr;

// Consumed by the `cfg(not(gpu))` CPU path in `processing::algorithms` and by
// tests (via `super::cpu::*`). Under `gpu` the non-test consumers vanish, so the
// lib build reports them unused; silence only there.
#[cfg_attr(feature = "gpu", allow(unused_imports))]
pub use das::delay_and_sum_cpu;
#[cfg_attr(feature = "gpu", allow(unused_imports))]
pub use mvdr::mvdr_cpu;
