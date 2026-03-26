//! Electromagnetic physics traits and interfaces
//!
//! This module defines the trait-based interfaces for electromagnetic wave
//! physics, including Maxwell's equations, photoacoustic coupling, and
//! plasmonic effects.

pub mod maxwell;
pub mod photoacoustic;
pub mod plasmonic;
pub mod source;

pub use maxwell::ElectromagneticWaveEquation;
pub use photoacoustic::PhotoacousticCoupling;
pub use plasmonic::PlasmonicEnhancement;
pub use source::EMSource;

#[cfg(test)]
mod tests;
