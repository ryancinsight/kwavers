//! Time-of-Arrival Trilateration for Source Localization
//!
//! Implements trilateration algorithms that determine source position
//! from time-of-arrival (TOA) measurements at multiple sensors.
//!
//! # References
//!
//! - Foy, W. H. (1976). "Position-Location Solutions by Taylor-Series Estimation"
//! - Smith & Abel (1987). "Closed-Form Least-Squares Source Location from Range Differences"

pub mod solver;
#[cfg(test)]
mod tests;
pub mod types;

pub use solver::Trilateration;
pub use types::{LocalizationResult, TrilaterationConfig};
