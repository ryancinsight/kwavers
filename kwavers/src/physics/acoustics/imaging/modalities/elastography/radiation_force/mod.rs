//! Acoustic Radiation Force for Shear Wave Generation
//!
//! Implements acoustic radiation force impulse (ARFI) for generating shear waves
//! in soft tissue.
//!
//! ## Physics
//!
//! Acoustic radiation force arises from momentum transfer when ultrasound waves
//! are absorbed or reflected. For a focused ultrasound beam:
//!
//! F = (2αI)/c
//!
//! where:
//! - F is radiation force density (N/m³)
//! - α is absorption coefficient (Np/m)
//! - I is acoustic intensity (W/m²)
//! - c is sound speed (m/s)
//!
//! ## References
//!
//! - Nightingale, K., et al. (2002). "Acoustic radiation force impulse imaging."
//!   *Ultrasound in Medicine & Biology*, 28(2), 227-235.
//! - Palmeri, M. L., et al. (2005). "Ultrasonic tracking of acoustic radiation
//!   force-induced displacements." *IEEE TUFFC*, 52(8), 1300-1313.

pub mod impulse;
pub mod patterns;
pub mod tracking;

#[cfg(test)]
mod tests;

pub use impulse::{AcousticRadiationForce, PushPulseParameters};
pub use patterns::{MultiDirectionalPush, DirectionalPush};
pub use tracking::{DirectionalWaveTracker, TrackingRegion, DirectionalQuality, ValidationResult};
