//! Electromagnetic physics traits and interfaces
//!
//! This module defines the trait-based interfaces for electromagnetic wave
//! physics, including Maxwell's equations, photoacoustic coupling, and
//! plasmonic effects.

/// Maxwell's-equations wave specification trait.
pub mod maxwell;
/// Photoacoustic (optical-absorption → acoustic-source) coupling trait.
pub mod photoacoustic;
/// Plasmonic-enhancement (nanoparticle resonance) trait.
pub mod plasmonic;
/// Electromagnetic source-term specification trait.
pub mod source;

pub use maxwell::ElectromagneticWaveEquation;
pub use photoacoustic::PhotoacousticCoupling;
pub use plasmonic::PlasmonicEnhancementEquation;
pub use source::PhysicsEMSource;

#[cfg(test)]
mod tests;
