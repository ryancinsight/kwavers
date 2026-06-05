//! FDTD Reference Solution Generator for PINN Validation
//!
//! Provides finite-difference time-domain (FDTD) reference solutions
//! for the 1D wave equation to validate PINN predictions.
//!
//! ## Implementation
//!
//! Uses central difference scheme:
//! - Spatial: ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
//! - Temporal: ∂²u/∂t² ≈ (u[n+1] - 2u[n] + u[n-1]) / dt²
//!
//! ## References
//!
//! - Courant-Friedrichs-Lewy (CFL) condition: c×dt/dx ≤ 1

mod config;
mod solver;
#[cfg(test)]
mod tests;

pub use config::FDTDConfig;
pub use solver::FDTD1DWaveSolver;

/// Initial condition types for 1D wave equation
#[derive(Debug, Clone, Copy)]
pub enum InitialCondition {
    /// Gaussian pulse at center
    GaussianPulse { width: f64, amplitude: f64 },
    /// Sine wave
    SineWave { frequency: f64, amplitude: f64 },
    /// Custom (user-provided)
    Custom,
}
