//! Wave propagation physics: reflection, refraction, and transmission
//!
//! This module implements fundamental wave propagation phenomena at interfaces
//! between media with different properties, following Snell's law and Fresnel equations.
//!
//! # Literature References
//!
//! 1. **Born, M., & Wolf, E. (1999)**. "Principles of Optics" (7th ed.).
//!    Cambridge University Press. ISBN: 978-0521642224
//!    - Comprehensive treatment of reflection and refraction
//!    - Fresnel equations derivation and applications
//!
//! 2. **Kinsler, L. E., et al. (2000)**. "Fundamentals of Acoustics" (4th ed.).
//!    Wiley. ISBN: 978-0471847892
//!    - Acoustic reflection and transmission coefficients
//!    - Mode conversion at interfaces
//!
//! 3. **Brekhovskikh, L. M., & Lysanov, Y. P. (2003)**. "Fundamentals of Ocean Acoustics"
//!    (3rd ed.). Springer. ISBN: 978-0387954677
//!    - Reflection from layered media
//!    - Critical angles and total internal reflection
//!
//! 4. **Pierce, A. D. (2019)**. "Acoustics: An Introduction to Its Physical
//!    Principles and Applications" (3rd ed.). Springer. ISBN: 978-3030112134
//!    - Comprehensive acoustic wave propagation theory

// Core submodules
pub mod attenuation;
pub mod calculator;
pub mod coefficients;
pub mod fresnel;
pub mod interface;
pub mod medium;
pub mod reflection;
pub mod refraction;
pub mod scattering;
pub mod snell;

// Re-export core types
pub use attenuation::AttenuationCalculator;
pub use calculator::WavePropagationCalculator;
pub use coefficients::PropagationCoefficients;
pub use fresnel::{FresnelCalculator, FresnelCoefficients};
pub use interface::{Interface, InterfaceProperties, InterfaceType};
pub use medium::MediumProperties;
pub use reflection::{ReflectionCalculator, ReflectionCoefficients};
pub use refraction::{RefractionAngles, RefractionCalculator};
pub use scattering::{PhaseFunction, ScatteringCalculator, ScatteringRegime, VolumeScattering};
pub use snell::{CriticalAngles, SnellLawCalculator};

/// Wave propagation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveMode {
    /// Acoustic pressure wave
    Acoustic,
    /// Optical electromagnetic wave  
    Optical,
    /// Elastic shear wave
    ElasticShear,
    /// Elastic compressional wave
    ElasticCompressional,
}

/// Polarization state for electromagnetic waves
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Transverse electric (S-polarized)
    TransverseElectric,
    /// Transverse magnetic (P-polarized)
    TransverseMagnetic,
    /// Unpolarized
    Unpolarized,
    /// Circular polarization
    Circular,
    /// Elliptical polarization
    Elliptical,
}
