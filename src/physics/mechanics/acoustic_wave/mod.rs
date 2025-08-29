//! Acoustic wave mechanics module
//!
//! This module provides implementations for various acoustic wave propagation models
//! including linear and nonlinear wave equations.

// Submodules
pub mod kuznetsov;
pub mod nonlinear;
pub mod unified;
pub mod westervelt;
pub mod westervelt_fdtd;
pub mod westervelt_wave;

// Test support (only available in test builds)
#[cfg(test)]
mod test_support;

// Re-exports for convenience
pub use kuznetsov::{KuznetsovConfig, KuznetsovWave};
pub use nonlinear::NonlinearWave;
pub use unified::{AcousticModelType, AcousticSolverConfig, UnifiedAcousticSolver};
pub use westervelt::WesterveltWave;
pub use westervelt_fdtd::{WesterveltFdtd, WesterveltFdtdConfig};

use crate::grid::Grid;
use crate::medium::{acoustic::AcousticProperties, core::CoreMedium, Medium};
use std::f64::consts::PI;

// Physical constants
const ACOUSTIC_DIFFUSIVITY_COEFFICIENT: f64 = 2.0;

/// Compute acoustic diffusivity from medium properties
///
/// This is the single source of truth for acoustic diffusivity calculation.
///
/// # Physics Background
///
/// Acoustic diffusivity δ = (4μ/3 + μ_B + κ(γ-1)/C_p) / ρ₀
/// Where:
/// - μ = shear viscosity
/// - μ_B = bulk viscosity  
/// - κ = thermal conductivity
/// - γ = specific heat ratio
/// - C_p = specific heat at constant pressure
///
/// For soft tissues, we use the approximation:
/// δ ≈ 2αc³/(ω²)
///
/// where α is the absorption coefficient and c is the sound speed.
///
/// # Safety
///
/// Returns 0.0 for zero frequency (static fields) to prevent division by zero.
/// This is physically sensible as the frequency-dependent absorption model
/// becomes ill-defined at DC.
pub fn compute_acoustic_diffusivity<M: Medium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    frequency: f64,
    grid: &Grid,
) -> f64 {
    if frequency == 0.0 {
        return 0.0;
    }

    let alpha = medium.absorption_coefficient(x, y, z, grid, frequency);
    let c = medium.sound_speed(x, y, z, grid);
    let omega = 2.0 * PI * frequency;

    ACOUSTIC_DIFFUSIVITY_COEFFICIENT * alpha * c.powi(3) / (omega * omega)
}

/// Compute the maximum stable time step for acoustic wave propagation
///
/// Based on the CFL (Courant-Friedrichs-Lewy) condition for stability.
///
/// # Arguments
/// * `grid` - Computational grid
/// * `max_sound_speed` - Maximum sound speed in the medium
/// * `spatial_order` - Order of spatial discretization (2, 4, or 6)
///
/// # Returns
/// Maximum stable time step
pub fn compute_max_stable_timestep(grid: &Grid, max_sound_speed: f64, spatial_order: usize) -> f64 {
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);

    // CFL limits for different spatial orders
    let cfl_limit = match spatial_order {
        2 => 1.0 / (3.0_f64).sqrt(), // Theoretical limit: 1/√3 ≈ 0.577
        4 => 0.50,                   // Conservative value for 4th-order
        6 => 0.40,                   // Conservative value for 6th-order
        _ => 1.0 / (3.0_f64).sqrt(), // Default to 2nd-order limit
    };

    cfl_limit * min_dx / max_sound_speed
}

/// Compute nonlinearity coefficient for a given medium
///
/// The nonlinearity coefficient β = 1 + B/(2A) where B/A is the
/// nonlinearity parameter of the medium.
///
/// # Arguments
/// * `medium` - The acoustic medium
/// * `x`, `y`, `z` - Position coordinates
/// * `grid` - Computational grid
///
/// # Returns
/// Nonlinearity coefficient β
pub fn compute_nonlinearity_coefficient<M: Medium + ?Sized>(
    medium: &M,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
) -> f64 {
    let b_over_a = medium.nonlinearity_coefficient(x, y, z, grid);
    1.0 + b_over_a / 2.0
}
