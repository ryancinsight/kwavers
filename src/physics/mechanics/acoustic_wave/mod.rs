//! Acoustic wave mechanics module
//!
//! This module provides implementations for various acoustic wave propagation models
//! including linear and nonlinear wave equations.

// Submodules
pub mod kuznetsov;
pub mod kzk;
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
use crate::medium::{core::CoreMedium, Medium};
use std::f64::consts::PI;

// Physical constants
// Coefficient in the simplified acoustic diffusivity formula for soft tissues
// δ ≈ 2αc³/(ω²) where this constant represents the factor 2
const SOFT_TISSUE_DIFFUSIVITY_APPROXIMATION_FACTOR: f64 = 2.0;

/// Spatial discretization order for numerical schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialOrder {
    /// Second-order accurate spatial discretization
    Second,
    /// Fourth-order accurate spatial discretization
    Fourth,
    /// Sixth-order accurate spatial discretization
    Sixth,
}

impl SpatialOrder {
    /// Get the CFL stability limit for this spatial order
    #[must_use]
    pub fn cfl_limit(&self) -> f64 {
        match self {
            SpatialOrder::Second => 1.0 / (3.0_f64).sqrt(), // Theoretical limit: 1/√3 ≈ 0.577
            SpatialOrder::Fourth => 0.50,                   // Conservative value for 4th-order
            SpatialOrder::Sixth => 0.40,                    // Conservative value for 6th-order
        }
    }

    /// Get the minimum number of grid points required for this spatial order
    #[must_use]
    pub fn minimum_grid_points(&self) -> usize {
        match self {
            SpatialOrder::Second => 3,
            SpatialOrder::Fourth => 5,
            SpatialOrder::Sixth => 7,
        }
    }

    /// Convert from usize for backward compatibility
    #[must_use]
    pub fn from_usize(order: usize) -> Self {
        match order {
            2 => SpatialOrder::Second,
            4 => SpatialOrder::Fourth,
            6 => SpatialOrder::Sixth,
            _ => SpatialOrder::Second, // Default to second order
        }
    }
}

/// Compute acoustic diffusivity from medium properties
///
/// This is the single source of truth for acoustic diffusivity calculation.
///
/// # Physics Background
///
/// Acoustic diffusivity δ = (4μ/3 + `μ_B` + κ(γ-1)/C_p) / ρ₀
/// Where:
/// - μ = shear viscosity
/// - `μ_B` = bulk viscosity  
/// - κ = thermal conductivity
/// - γ = specific heat ratio
/// - `C_p` = specific heat at constant pressure
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

    let alpha =
        crate::medium::AcousticProperties::absorption_coefficient(medium, x, y, z, grid, frequency);
    let (i, j, k) = crate::medium::continuous_to_discrete(x, y, z, grid);
    let c = medium.sound_speed(i, j, k);
    let omega = 2.0 * PI * frequency;

    SOFT_TISSUE_DIFFUSIVITY_APPROXIMATION_FACTOR * alpha * c.powi(3) / (omega * omega)
}

/// Compute the maximum stable time step for acoustic wave propagation
///
/// Based on the CFL (Courant-Friedrichs-Lewy) condition for stability.
///
/// # Arguments
/// * `grid` - Computational grid
/// * `max_sound_speed` - Maximum sound speed in the medium
/// * `spatial_order` - Order of spatial discretization
///
/// # Returns
/// Maximum stable time step
pub fn compute_max_stable_timestep(
    grid: &Grid,
    max_sound_speed: f64,
    spatial_order: SpatialOrder,
) -> f64 {
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let cfl_limit = spatial_order.cfl_limit();
    cfl_limit * min_dx / max_sound_speed
}

/// Compute the maximum stable time step (backward compatibility wrapper)
///
/// # Deprecated
/// Use the version with `SpatialOrder` enum instead
#[deprecated(note = "Use compute_max_stable_timestep with SpatialOrder enum")]
pub fn compute_max_stable_timestep_usize(
    grid: &Grid,
    max_sound_speed: f64,
    spatial_order: usize,
) -> f64 {
    compute_max_stable_timestep(
        grid,
        max_sound_speed,
        SpatialOrder::from_usize(spatial_order),
    )
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
    let b_over_a =
        crate::medium::AcousticProperties::nonlinearity_coefficient(medium, x, y, z, grid);
    1.0 + b_over_a / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_order_cfl_limits() {
        assert!((SpatialOrder::Second.cfl_limit() - 0.577).abs() < 0.001);
        assert_eq!(SpatialOrder::Fourth.cfl_limit(), 0.50);
        assert_eq!(SpatialOrder::Sixth.cfl_limit(), 0.40);
    }

    #[test]
    fn test_spatial_order_minimum_points() {
        assert_eq!(SpatialOrder::Second.minimum_grid_points(), 3);
        assert_eq!(SpatialOrder::Fourth.minimum_grid_points(), 5);
        assert_eq!(SpatialOrder::Sixth.minimum_grid_points(), 7);
    }

    #[test]
    fn test_spatial_order_from_usize() {
        assert_eq!(SpatialOrder::from_usize(2), SpatialOrder::Second);
        assert_eq!(SpatialOrder::from_usize(4), SpatialOrder::Fourth);
        assert_eq!(SpatialOrder::from_usize(6), SpatialOrder::Sixth);
        assert_eq!(SpatialOrder::from_usize(99), SpatialOrder::Second); // Invalid defaults to Second
    }
}
