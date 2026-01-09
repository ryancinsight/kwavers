//! Wave propagation physics module

pub mod attenuation;
pub mod calculator;
pub mod coefficients;
pub mod heterogeneity;
pub mod interfaces;

pub mod scattering;

pub use crate::domain::medium::AnalyticalMediumProperties;
pub use attenuation::AttenuationCalculator;
pub use calculator::WavePropagationCalculator;
pub use coefficients::PropagationCoefficients;
pub use interfaces::{FresnelCalculator, Interface, InterfaceType, SnellLawCalculator};

/// Wave mode enumeration for different types of wave propagation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveMode {
    /// Acoustic wave mode
    Acoustic,
    /// Optical wave mode
    Optical,
    /// Elastic shear wave mode
    ElasticShear,
    /// Elastic compressional wave mode
    ElasticCompressional,
    /// Longitudinal wave (compressional wave)
    Longitudinal,
    /// Transverse wave (shear wave)
    Transverse,
    /// Surface wave (Rayleigh wave)
    Surface,
    /// Plate wave (Lamb wave)
    Plate,
}

/// Polarization states for electromagnetic waves
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Transverse Electric (TE) polarization - electric field perpendicular to plane of incidence
    TransverseElectric,
    /// Transverse Magnetic (TM) polarization - magnetic field perpendicular to plane of incidence
    TransverseMagnetic,
    /// Unpolarized light - random polarization
    Unpolarized,
    /// Circular polarization - electric field rotates in a circular motion
    Circular,
    /// Elliptical polarization - electric field describes an ellipse
    Elliptical,
}
