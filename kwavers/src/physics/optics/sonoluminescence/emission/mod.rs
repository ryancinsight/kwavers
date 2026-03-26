//! Main sonoluminescence emission module
//!
//! Integrates blackbody, bremsstrahlung, and molecular emission models
//!
//! ## Physical Models
//!
//! ### Bubble Dynamics and Thermodynamics
//! This module implements the Rayleigh-Plesset equation with Keller-Miksis
//! compressible corrections for bubble wall motion, coupled with adiabatic
//! compression heating for temperature evolution.
//!
//! **Key Assumptions:**
//! - Adiabatic compression: T ∝ R^(3(1-γ)) where γ is the polytropic index
//! - Ideal gas behavior for bubble contents
//! - Spherical bubble geometry
//! - No heat conduction losses (adiabatic approximation)
//!
//! **Limitations:**
//! - Adiabatic approximation breaks down at extreme compression ratios
//! - Neglects thermal conduction and mass transfer effects
//! - Single-bubble approximation (no bubble-bubble interactions)
//!   TODO_AUDIT: P1 - Quantum Emission Models - Implement full quantum mechanical bremsstrahlung and Cherenkov radiation with relativistic corrections, replacing classical approximations
//!
//! **References:**
//! - Prosperetti (1991): "Bubble dynamics in a compressible liquid"
//! - Keller & Miksis (1980): "Bubble oscillations of large amplitude"
//! - Brenner et al. (2002): "Single-bubble sonoluminescence"
//!
//! ### Emission Models
//! - **Blackbody Radiation**: Planck's law with Wien and Rayleigh-Jeans approximations
//! - **Bremsstrahlung**: Free-free emission from ionized gas
//! - **Molecular Lines**: Placeholder for future implementation
//!
//! **Numerical Stability:**
//! - RK4 integration for bubble dynamics with adaptive timestep control
//! - CFL-like condition monitoring for compressibility effects
//! - Boundary condition enforcement (positive radius, reasonable temperatures)

pub mod orchestrator;
pub mod spectrum;
pub mod statistics;

#[cfg(test)]
mod tests;

pub use orchestrator::{IntegratedSonoluminescence, SonoluminescenceEmission};
pub use spectrum::{EmissionParameters, SpectralField};
pub use statistics::{SonoluminescencePulse, SpectralStatistics};
