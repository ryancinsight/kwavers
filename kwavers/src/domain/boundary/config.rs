//! Boundary condition configuration

use serde::{Deserialize, Serialize};

/// Boundary condition parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryParameters {
    /// Boundary type for each face [`x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`]
    pub boundary_types: [BoundaryType; 6],
    /// PML layer thickness in grid points
    pub pml_thickness: usize,
    /// PML absorption coefficient
    pub pml_alpha: f64,
}

/// Types of boundary conditions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BoundaryType {
    /// Perfectly matched layer
    PML,
    /// Periodic boundary
    Periodic,
    /// Dirichlet (fixed value)
    Dirichlet,
    /// Neumann (fixed gradient)
    Neumann,
    /// Absorbing boundary
    Absorbing,
    /// Reflecting boundary
    Reflecting,
}

impl BoundaryParameters {
    /// Validate boundary parameters
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if self.pml_thickness == 0 {
            for boundary in &self.boundary_types {
                if *boundary == BoundaryType::PML {
                    return Err(crate::core::error::ConfigError::InvalidValue {
                        parameter: "pml_thickness".to_string(),
                        value: "0".to_string(),
                        constraint: "Must be positive when using PML boundaries".to_string(),
                    }
                    .into());
                }
            }
        }

        if self.pml_alpha < 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "pml_alpha".to_string(),
                value: self.pml_alpha.to_string(),
                constraint: "Must be non-negative".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for BoundaryParameters {
    fn default() -> Self {
        Self {
            boundary_types: [BoundaryType::PML; 6],
            pml_thickness: 10,
            pml_alpha: 2.0,
        }
    }
}
