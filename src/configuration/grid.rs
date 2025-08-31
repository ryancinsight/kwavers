//! Grid discretization parameters

use serde::{Deserialize, Serialize};

/// Grid discretization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridParameters {
    /// Grid dimensions [nx, ny, nz]
    pub dimensions: [usize; 3],
    /// Grid spacing [dx, dy, dz] in meters
    pub spacing: [f64; 3],
    /// Grid origin [x0, y0, z0] in meters
    pub origin: [f64; 3],
    /// Staggered grid for velocity
    pub staggered: bool,
    /// Grid type
    pub grid_type: GridType,
}

/// Types of computational grids
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GridType {
    /// Uniform Cartesian grid
    Cartesian,
    /// Cylindrical coordinates
    Cylindrical,
    /// Spherical coordinates
    Spherical,
    /// Adaptive mesh refinement
    Adaptive,
}

impl GridParameters {
    /// Validate grid parameters
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        // Check dimensions
        for (i, &dim) in self.dimensions.iter().enumerate() {
            if dim == 0 {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: format!("dimensions[{}]", i),
                    value: dim.to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        // Check spacing
        for (i, &dx) in self.spacing.iter().enumerate() {
            if dx <= 0.0 {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: format!("spacing[{}]", i),
                    value: dx.to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        // Warn if grid is too large
        let total_points = self.dimensions[0] * self.dimensions[1] * self.dimensions[2];
        if total_points > 1_000_000_000 {
            log::warn!(
                "Grid contains {} points - may require significant memory",
                total_points
            );
        }

        Ok(())
    }

    /// Calculate total number of grid points
    pub fn total_points(&self) -> usize {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Calculate domain size in meters
    pub fn domain_size(&self) -> [f64; 3] {
        [
            self.dimensions[0] as f64 * self.spacing[0],
            self.dimensions[1] as f64 * self.spacing[1],
            self.dimensions[2] as f64 * self.spacing[2],
        ]
    }
}

impl Default for GridParameters {
    fn default() -> Self {
        Self {
            dimensions: [128, 128, 128],
            spacing: [1e-4, 1e-4, 1e-4], // 0.1mm
            origin: [0.0, 0.0, 0.0],
            staggered: false,
            grid_type: GridType::Cartesian,
        }
    }
}
