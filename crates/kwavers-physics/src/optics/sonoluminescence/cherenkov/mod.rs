//! Cherenkov Radiation Model for Sonoluminescence
//!
//! This module provides a physically motivated, threshold-based Cherenkov radiation model
//! usable by the sonoluminescence emission pipeline. It implements:
//! - Threshold condition `v > c/n`
//! - Angle computation `θ = arccos(1/(nβ))`
//! - Spectral intensity with inverse frequency scaling for numerical stability
//! - Dynamic refractive index updates with compression and temperature effects
//! - Field emission helper integrated with the main emission module
//!
//! References: Frank & Tamm (1937), Jackson (1999).

pub mod emission;
pub mod model;

pub use emission::calculate_cherenkov_emission;
pub use model::CherenkovModel;

#[cfg(test)]
mod tests;
