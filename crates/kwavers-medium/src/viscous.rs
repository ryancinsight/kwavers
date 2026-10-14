//! Viscous properties trait for fluid dynamics
//!
//! This module defines traits for viscous properties including shear and bulk
//! viscosity. Quantities are typed with their SI dimensions (ADR 0050's typed
//! exchange discipline): dynamic and bulk viscosity in Pa·s, kinematic
//! viscosity in m²/s.

use crate::core::CoreMedium;
use aequitas::systems::si::quantities::{DynamicViscosity, KinematicViscosity};
use kwavers_grid::Grid;

/// Trait for viscous medium properties
pub trait ViscousProperties: CoreMedium {
    /// Get dynamic viscosity (Pa·s)
    /// This is the general viscosity coefficient for simple fluids
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> DynamicViscosity;

    /// Get shear viscosity (Pa·s)
    /// Dynamic viscosity for Newtonian fluids
    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> DynamicViscosity {
        // For simple fluids, shear viscosity equals dynamic viscosity
        self.viscosity(x, y, z, grid)
    }

    /// Get bulk viscosity (Pa·s)
    /// Related to volume changes and compressibility
    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> DynamicViscosity {
        // Stokes' hypothesis: bulk viscosity = 2.5 * shear viscosity
        DynamicViscosity::from_base(2.5 * self.shear_viscosity(x, y, z, grid).into_base())
    }

    /// Get kinematic viscosity (m²/s)
    /// Ratio of dynamic viscosity to density
    fn kinematic_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> KinematicViscosity {
        let mu = self.viscosity(x, y, z, grid).into_base();
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        let rho = self.density(i, j, k);
        if rho > 0.0 {
            KinematicViscosity::from_base(mu / rho)
        } else {
            KinematicViscosity::from_base(0.0)
        }
    }
}
