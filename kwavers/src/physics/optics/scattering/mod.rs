// physics/optics/scattering/mod.rs
//! Mie scattering theory for electromagnetic wave scattering by spherical particles
//!
//! This module implements Mie scattering theory for calculating scattering and absorption
//! cross-sections of spherical particles. This is fundamental for understanding light
//! propagation in biological tissues and optical imaging applications.

pub mod calculator;
pub mod constants;
pub mod parameters;
pub mod rayleigh;
pub mod result;

#[cfg(test)]
mod tests;

pub use calculator::MieCalculator;
pub use parameters::MieParameters;
pub use rayleigh::RayleighScattering;
pub use result::MieResult;
