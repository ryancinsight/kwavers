//! Acoustic wave mechanics module
//!
//! This module provides implementations for various acoustic wave propagation models
//! including linear and nonlinear wave equations.

// Submodules
pub mod nonlinear;

// Test support (only available in test builds)
#[cfg(test)]
mod test_support;

// Re-exports for convenience
pub use nonlinear::NonlinearWave;

use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
#[allow(unused_imports)]
use core::prelude::rust_2024::derive;
use std::f64::consts::PI;

// Physical constants
// Coefficient relating power-law absorption to acoustic diffusivity for soft tissues
// Formula: δ ≈ 2αc³/(ω²) where α is frequency-dependent absorption
// Reference: Szabo (1995) "Time domain wave equations for lossy media" Eq. 14
// This constant represents the factor 2 in the relationship
const POWER_LAW_ABSORPTION_TO_DIFFUSIVITY_FACTOR: f64 = 2.0;

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
    ///
    /// For 3D finite difference schemes with central differences:
    /// - 2nd order: CFL ≤ 1/√(3) ≈ 0.577
    /// - 4th order: CFL ≤ 1/√(15) ≈ 0.258
    /// - 6th order: CFL ≤ 1/√(27) ≈ 0.192
    ///
    /// Reference: Gustafsson et al. (1995) "Time compact difference schemes"
    #[must_use]
    pub fn cfl_limit(&self) -> f64 {
        match self {
            SpatialOrder::Second => 1.0 / (3.0_f64).sqrt(), // 1/√3 ≈ 0.577
            SpatialOrder::Fourth => 1.0 / (15.0_f64).sqrt(), // 1/√15 ≈ 0.258
            SpatialOrder::Sixth => 1.0 / (27.0_f64).sqrt(), // 1/√27 ≈ 0.192
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

    /// Convert from usize, returning an error for invalid orders
    pub fn from_usize(order: usize) -> Result<Self, crate::core::error::KwaversError> {
        match order {
            2 => Ok(SpatialOrder::Second),
            4 => Ok(SpatialOrder::Fourth),
            6 => Ok(SpatialOrder::Sixth),
            _ => Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "spatial_order".to_string(),
                value: order.to_string(),
                constraint: "must be 2, 4, or 6".to_string(),
            }
            .into()),
        }
    }
}

/// Computes acoustic diffusivity from power-law absorption, an approximation valid for many biological tissues
///
/// This is the single source of truth for acoustic diffusivity calculation using
/// the power-law absorption model commonly used for soft tissues.
///
/// # Physics Background
///
/// For biological soft tissues, acoustic diffusivity can be approximated as:
/// δ ≈ 2αc³/(ω²)
///
/// where:
/// - α is the absorption coefficient
/// - c is the sound speed  
/// - ω is the angular frequency
///
/// This approximation is valid for many biological tissues but should not be used
/// for materials like water, bone, or industrial materials where the full viscosity
/// and thermal conduction terms are necessary.
///
/// For the complete formula:
/// δ = (4μ/3 + μ_B + κ(γ-1)/C_p) / ρ₀
/// Where:
/// - μ = shear viscosity
/// - μ_B = bulk viscosity  
/// - κ = thermal conductivity
/// - γ = specific heat ratio
/// - C_p = specific heat at constant pressure
/// - ρ₀ = reference density
///
/// # Safety
///
/// Returns 0.0 for zero frequency (static fields) to prevent division by zero.
/// This is physically sensible as the frequency-dependent absorption model
/// becomes ill-defined at DC.
pub fn compute_diffusivity_from_power_law_absorption<M: Medium + ?Sized>(
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

    let alpha = crate::domain::medium::AcousticProperties::absorption_coefficient(
        medium, x, y, z, grid, frequency,
    );
    let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
    let c = medium.sound_speed(i, j, k);
    let omega = 2.0 * PI * frequency;

    POWER_LAW_ABSORPTION_TO_DIFFUSIVITY_FACTOR * alpha * c.powi(3) / (omega * omega)
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
        crate::domain::medium::AcousticProperties::nonlinearity_coefficient(medium, x, y, z, grid);
    1.0 + b_over_a / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use core::prelude::rust_2024::test;
    #[allow(unused_imports)]
    use core::{assert, assert_eq};

    #[test]
    fn test_spatial_order_cfl_limits() {
        // CORRECTED: Theoretical CFL limits for 3D finite difference schemes
        assert!((SpatialOrder::Second.cfl_limit() - 0.577).abs() < 0.001); // 1/√3
        assert!((SpatialOrder::Fourth.cfl_limit() - 0.258).abs() < 0.001); // 1/√15
        assert!((SpatialOrder::Sixth.cfl_limit() - 0.192).abs() < 0.001); // 1/√27
    }

    #[test]
    fn test_spatial_order_minimum_points() {
        assert_eq!(SpatialOrder::Second.minimum_grid_points(), 3);
        assert_eq!(SpatialOrder::Fourth.minimum_grid_points(), 5);
        assert_eq!(SpatialOrder::Sixth.minimum_grid_points(), 7);
    }

    #[test]
    fn test_spatial_order_from_usize() {
        assert_eq!(SpatialOrder::from_usize(2).unwrap(), SpatialOrder::Second);
        assert_eq!(SpatialOrder::from_usize(4).unwrap(), SpatialOrder::Fourth);
        assert_eq!(SpatialOrder::from_usize(6).unwrap(), SpatialOrder::Sixth);
        assert!(SpatialOrder::from_usize(99).is_err()); // Invalid returns error
    }
}
