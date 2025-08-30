//! Wave propagation physics module
//!
//! Implements reflection, refraction, transmission, and scattering
//! at interfaces between media with different properties.
//!
//! # Literature References
//!
//! - Born & Wolf (1999): "Principles of Optics" (7th ed.)
//! - Kinsler et al. (2000): "Fundamentals of Acoustics" (4th ed.)
//! - Pierce (2019): "Acoustics: Physical Principles and Applications" (3rd ed.)

// Core types
pub mod types;

// Physics implementations
pub mod attenuation;
pub mod calculator;
pub mod fresnel;
pub mod interface;
pub mod medium_properties;
pub mod reflection;
pub mod refraction;
pub mod scattering;
pub mod snell;

// Re-export primary types
pub use attenuation::AttenuationCalculator;
pub use calculator::{InterfaceCoefficients, InterfaceResponse, WavePropagationCalculator};
pub use medium_properties::MediumProperties;
pub use types::{Interface, Polarization, WaveMode};

// Re-export specialized calculators
pub use fresnel::{FresnelCalculator, FresnelCoefficients};
pub use interface::{InterfaceProperties, InterfaceType};
pub use reflection::{ReflectionCalculator, ReflectionCoefficients};
pub use refraction::{RefractionAngles, RefractionCalculator};
pub use scattering::{PhaseFunction, ScatteringCalculator, ScatteringRegime, VolumeScattering};
pub use snell::{CriticalAngles, SnellLawCalculator};
