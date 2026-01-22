//! Staircase Boundary Smoothing
//!
//! Reduces grid artifacts at curved boundaries by smoothing the staircase
//! representation inherent to Cartesian grids. This improves accuracy for
//! problems with complex geometries (curved vessels, focused transducers, etc.).
//!
//! # Problem Statement
//!
//! Cartesian grids approximate curved boundaries as staircases, leading to:
//! - **Spurious reflections** from grid edges
//! - **Numerical dispersion** errors
//! - **Reduced accuracy** near boundaries
//!
//! # Solution Approaches
//!
//! ## 1. Immersed Interface Method (IIM)
//! - Modify finite-difference stencils near boundaries
//! - Jump conditions incorporated into discretization
//! - Maintains second-order accuracy
//!
//! ## 2. Ghost Cell Method
//! - Extrapolate values into ghost cells beyond boundary
//! - Smooth representation without grid refinement
//! - Compatible with existing FDTD/PSTD solvers
//!
//! ## 3. Subgrid Averaging
//! - Average material properties over grid cells intersecting boundary
//! - Volume-weighted averaging for acoustic impedance
//! - Simple to implement, effective for mild curvature
//!
//! # Literature References
//!
//! - LeVeque, R.J. & Li, Z. (1994). "The immersed interface method for elliptic equations with discontinuous coefficients and singular sources". *SIAM J. Numer. Anal.*, 31(4), 1019-1044.
//! - Mittal, R. & Iaccarino, G. (2005). "Immersed boundary methods". *Annual Review of Fluid Mechanics*, 37, 239-261.
//! - Treeby, B.E. et al. (2012). "Modeling nonlinear ultrasound propagation in heterogeneous media with power law absorption using a k-space pseudospectral method". *J. Acoust. Soc. Am.*, 131(6), 4324-4336.

pub mod ghost_cell;
pub mod immersed_interface;
pub mod subgrid;

pub use ghost_cell::{GhostCellConfig, GhostCellMethod};
pub use immersed_interface::{IIMConfig, ImmersedInterfaceMethod, JumpConditionType};
pub use subgrid::{SubgridAveraging, SubgridConfig};

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Boundary smoothing method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingMethod {
    /// No smoothing (standard staircase)
    None,
    /// Subgrid averaging (simple, effective)
    Subgrid,
    /// Ghost cell extrapolation
    GhostCell,
    /// Immersed interface method (most accurate)
    ImmersedInterface,
}

/// Boundary smoothing configuration
#[derive(Debug, Clone)]
pub struct BoundarySmoothingConfig {
    /// Smoothing method to use
    pub method: SmoothingMethod,

    /// Subgrid averaging parameters
    pub subgrid: Option<SubgridConfig>,

    /// Ghost cell parameters
    pub ghost_cell: Option<GhostCellConfig>,

    /// Immersed interface parameters
    pub iim: Option<IIMConfig>,
}

impl Default for BoundarySmoothingConfig {
    fn default() -> Self {
        Self {
            method: SmoothingMethod::Subgrid,
            subgrid: Some(SubgridConfig::default()),
            ghost_cell: None,
            iim: None,
        }
    }
}

/// Boundary smoothing processor
///
/// Applies smoothing to reduce staircase artifacts at curved boundaries.
#[derive(Debug, Clone)]
pub struct BoundarySmoothing {
    config: BoundarySmoothingConfig,
}

impl BoundarySmoothing {
    /// Create a new boundary smoothing processor
    pub fn new(config: BoundarySmoothingConfig) -> Self {
        Self { config }
    }

    /// Apply smoothing to material property field
    ///
    /// # Arguments
    ///
    /// * `property` - Material property field (e.g., sound speed, density)
    /// * `geometry` - Geometry indicator (1.0 inside, 0.0 outside)
    ///
    /// # Returns
    ///
    /// Smoothed property field
    pub fn smooth(
        &self,
        property: &Array3<f64>,
        geometry: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        match self.config.method {
            SmoothingMethod::None => {
                // No smoothing, return original
                Ok(property.clone())
            }
            SmoothingMethod::Subgrid => {
                let config = self.config.subgrid.clone().unwrap_or_default();
                let smoother = SubgridAveraging::new(config);
                smoother.apply(property, geometry)
            }
            SmoothingMethod::GhostCell => {
                let config = self.config.ghost_cell.clone().unwrap_or_default();
                let smoother = GhostCellMethod::new(config);
                smoother.apply(property, geometry)
            }
            SmoothingMethod::ImmersedInterface => {
                let config = self.config.iim.clone().unwrap_or_default();
                let smoother = ImmersedInterfaceMethod::new(config);
                smoother.apply(property, geometry)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_no_smoothing() {
        let config = BoundarySmoothingConfig {
            method: SmoothingMethod::None,
            ..Default::default()
        };

        let smoother = BoundarySmoothing::new(config);

        let property = Array3::from_elem((10, 10, 10), 1540.0);
        let geometry = Array3::from_elem((10, 10, 10), 1.0);

        let result = smoother.smooth(&property, &geometry);
        assert!(result.is_ok());

        let smoothed = result.unwrap();
        assert_eq!(smoothed, property);
    }
}
