//! Plane wave imaging and delay calculation.
//!
//! Implements geometric delay calculation for plane wave ultrasound imaging,
//! supporting tilted plane wave transmission and coherent compounding.

pub mod config;
pub mod processor;
#[cfg(test)]
mod tests;

pub use config::UltrafastPlaneWaveConfig;
pub use processor::UltrafastPlaneWave;
