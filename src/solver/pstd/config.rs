//! PSTD solver configuration

use crate::error::ValidationError;
use crate::validation::{Validatable, ValidationResult};
use serde::{Deserialize, Serialize};

/// PSTD solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PstdConfig {
    /// Enable k-space correction for improved accuracy
    pub use_kspace_correction: bool,

    /// k-space correction method
    pub correction_method: CorrectionMethod,

    /// Enable anti-aliasing (2/3 rule)
    pub use_antialiasing: bool,

    /// Enable absorption
    pub use_absorption: bool,

    /// Power law absorption parameters
    pub absorption_alpha: f64,
    pub absorption_y: f64,

    /// CFL safety factor (typically 0.3-0.5 for PSTD)
    pub cfl_factor: f64,

    /// Maximum number of time steps
    pub max_steps: usize,

    /// Enable dispersive media support
    pub dispersive_media: bool,

    /// PML boundary layers (if using PML)
    pub pml_layers: usize,

    /// PML absorption coefficient
    pub pml_alpha: f64,
}

/// k-space correction method
#[derive(Debug, Clone, Copy, Serialize, Deserialize))]
pub enum CorrectionMethod {
    None,
    Sinc,
    Exact,
    Modified,
}

impl Default for PstdConfig {
    fn default() -> Self {
        Self {
            use_kspace_correction: true,
            correction_method: CorrectionMethod::Sinc,
            use_antialiasing: true,
            use_absorption: false,
            absorption_alpha: 0.0,
            absorption_y: 1.0,
            cfl_factor: 0.3,
            max_steps: 1000,
            dispersive_media: false,
            pml_layers: 20,
            pml_alpha: 2.0,
        }
    }
}

impl Validatable for PstdConfig {
    fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();

        if self.cfl_factor <= 0.0 || self.cfl_factor > 1.0 {
            errors.push(ValidationError::FieldValidation {
                field: "cfl_factor".to_string(),
                value: self.cfl_factor.to_string(),
                constraint: "Must be in (0, 1]".to_string(),
            });
        }

        if self.absorption_y < 0.0 || self.absorption_y > 3.0 {
            errors.push(ValidationError::FieldValidation {
                field: "absorption_y".to_string(),
                value: self.absorption_y.to_string(),
                constraint: "Must be in [0, 3]".to_string(),
            });
        }

        if self.pml_layers > 50 {
            errors.push(ValidationError::FieldValidation {
                field: "pml_layers".to_string(),
                value: self.pml_layers.to_string(),
                constraint: "Must be <= 50".to_string(),
            });
        }

        ValidationResult::from_errors(errors.into_iter().map(|e| e.to_string()).collect())
    }
}
