//! Viscous properties trait for fluid dynamics
//!
//! This module defines traits for viscous properties including shear and bulk viscosity.

use crate::domain::grid::Grid;
use crate::domain::medium::core::CoreMedium;

/// Trait for viscous medium properties
pub trait ViscousProperties: CoreMedium {
    /// Get dynamic viscosity (Pa·s)
    /// This is the general viscosity coefficient for simple fluids
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get shear viscosity (Pa·s)
    /// Dynamic viscosity for Newtonian fluids
    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // For simple fluids, shear viscosity equals dynamic viscosity
        self.viscosity(x, y, z, grid)
    }

    /// Get bulk viscosity (Pa·s)
    /// Related to volume changes and compressibility
    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Stokes' hypothesis: bulk viscosity = 2.5 * shear viscosity
        2.5 * self.shear_viscosity(x, y, z, grid)
    }

    /// Get kinematic viscosity (m²/s)
    /// Ratio of dynamic viscosity to density
    fn kinematic_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.viscosity(x, y, z, grid);
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let rho = self.density(i, j, k);
        if rho > 0.0 {
            mu / rho
        } else {
            0.0
        }
    }
}
