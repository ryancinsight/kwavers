//! FDTD solver configuration

use crate::error::ValidationError;
use crate::physics::constants::numerical::CFL_SAFETY_FACTOR;
use crate::ValidationResult;
use serde::{Deserialize, Serialize};

/// FDTD solver configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FdtdConfig {
    /// Spatial derivative order (2, 4, or 6)
    pub spatial_order: usize,
    /// Use staggered grid (Yee cell)
    pub staggered_grid: bool,
    /// CFL safety factor (typically 0.3 for 3D FDTD)
    pub cfl_factor: f64,
    /// Enable subgridding for local refinement
    pub subgridding: bool,
    /// Subgridding refinement factor
    pub subgrid_factor: usize,
    /// Enable GPU acceleration (requires "gpu" feature)
    pub enable_gpu_acceleration: bool,
}

impl Default for FdtdConfig {
    fn default() -> Self {
        Self {
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: CFL_SAFETY_FACTOR,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
        }
    }
}

impl FdtdConfig {
    #[must_use]
    pub fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        // Validate spatial order
        if ![2, 4, 6].contains(&self.spatial_order) {
            errors.push(ValidationError::FieldValidation {
                field: "spatial_order".to_string(),
                value: self.spatial_order.to_string(),
                constraint: "Must be 2, 4, or 6".to_string(),
            });
        }

        // Validate CFL factor (max stable for 3D is 1/sqrt(3) ≈ 0.577)
        if self.cfl_factor <= 0.0 || self.cfl_factor > 0.577 {
            errors.push(ValidationError::FieldValidation {
                field: "cfl_factor".to_string(),
                value: self.cfl_factor.to_string(),
                constraint: "Must be in (0, 0.577] for 3D stability".to_string(),
            });
        }

        // Validate subgridding
        if self.subgridding && self.subgrid_factor < 2 {
            errors.push(ValidationError::FieldValidation {
                field: "subgrid_factor".to_string(),
                value: self.subgrid_factor.to_string(),
                constraint: "Must be >= 2".to_string(),
            });
        }

        if errors.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::from_errors(errors.into_iter().map(|e| e.to_string()).collect())
        }
    }
}
