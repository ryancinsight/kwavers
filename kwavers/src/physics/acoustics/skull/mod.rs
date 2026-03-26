//! Transcranial Focused Ultrasound (tFUS) - Skull Heterogeneity Modeling
//!
//! This module implements skull propagation and aberration correction for
//! transcranial ultrasound applications following clinical standards.
//!
//! ## Overview
//!
//! Skull bone introduces significant challenges for transcranial ultrasound:
//! 1. **High Acoustic Impedance**: Causes strong reflections (~80%)
//! 2. **Phase Aberrations**: Spatially varying thickness and density
//! 3. **Attenuation**: Frequency-dependent energy loss
//! 4. **Shear Wave Conversion**: Mode conversion at interfaces
//!
//! ## Literature References
//!
//! - Clement, G. T., & Hynynen, K. (2002). "A non-invasive method for focusing
//!   ultrasound through the skull." *Physics in Medicine & Biology*, 47(8), 1219.
//! - Aubry, J. F., et al. (2003). "Experimental demonstration of noninvasive
//!   transskull adaptive focusing." *IEEE TUFFC*, 50(10), 1128-1138.
//! - Marquet, F., et al. (2009). "Non-invasive transcranial ultrasound therapy
//!   based on a 3D CT scan." *Physics in Medicine & Biology*, 54(9), 2597.
//! - Pinton, G., et al. (2012). "Attenuation, scattering, and absorption of
//!   ultrasound in the skull bone." *Medical Physics*, 39(1), 299-307.

pub mod aberration;
pub mod attenuation;
pub mod heterogeneous;

pub mod analytical;
pub mod properties;
pub mod simulation;

#[cfg(test)]
mod tests;

pub use aberration::AberrationCorrection;
pub use attenuation::SkullAttenuation;
pub use heterogeneous::HeterogeneousSkull;
pub use properties::SkullProperties;
pub use simulation::TranscranialSimulation;
