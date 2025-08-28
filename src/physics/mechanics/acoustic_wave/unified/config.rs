//! Configuration for unified acoustic solver

use crate::error::{KwaversResult, ValidationError};

/// Type of acoustic model to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcousticModelType {
    /// Linear acoustic wave equation
    Linear,
    /// Westervelt equation for nonlinear acoustics in thermoviscous fluids
    Westervelt,
    /// Kuznetsov equation for nonlinear acoustics with diffusion
    Kuznetsov,
}

/// Configuration for the unified acoustic solver
#[derive(Debug, Clone)]
pub struct AcousticSolverConfig {
    /// Type of acoustic model to use
    pub model_type: AcousticModelType,

    /// Enable k-space correction for numerical dispersion
    pub k_space_correction: bool,

    /// Order of k-space correction (1 or 2)
    pub k_space_order: usize,

    /// Nonlinearity scaling factor (B/A parameter related)
    pub nonlinearity_scaling: f64,

    /// Maximum allowed pressure for stability
    pub max_pressure: f64,

    /// CFL safety factor for stability
    pub cfl_safety_factor: f64,

    /// Enable adaptive time stepping
    pub adaptive_time_stepping: bool,

    /// Minimum time step for adaptive stepping
    pub min_dt: f64,

    /// Maximum time step for adaptive stepping
    pub max_dt: f64,
}

impl Default for AcousticSolverConfig {
    fn default() -> Self {
        Self {
            model_type: AcousticModelType::Linear,
            k_space_correction: true,
            k_space_order: 2,
            nonlinearity_scaling: 3.5, // B/A for water at 20°C
            max_pressure: 1e7,          // 10 MPa
            cfl_safety_factor: 0.5,
            adaptive_time_stepping: false,
            min_dt: 1e-12,
            max_dt: 1e-6,
        }
    }
}

impl AcousticSolverConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.k_space_order != 1 && self.k_space_order != 2 {
            return Err(ValidationError::RangeValidation {
                field: "k_space_order".to_string(),
                value: self.k_space_order.to_string(),
                min: "1".to_string(),
                max: "2".to_string(),
            }.into());
        }

        if self.cfl_safety_factor <= 0.0 || self.cfl_safety_factor > 1.0 {
            return Err(ValidationError::RangeValidation {
                field: "cfl_safety_factor".to_string(),
                value: self.cfl_safety_factor.to_string(),
                min: "0.0 (exclusive)".to_string(),
                max: "1.0".to_string(),
            }.into());
        }

        if self.adaptive_time_stepping && self.min_dt >= self.max_dt {
            return Err(ValidationError::FieldValidation {
                field: "adaptive_time_stepping".to_string(),
                value: format!("min_dt={}, max_dt={}", self.min_dt, self.max_dt),
                constraint: "min_dt must be less than max_dt".to_string(),
            }.into());
        }

        Ok(())
    }

    /// Create configuration for linear acoustics
    pub fn linear() -> Self {
        Self {
            model_type: AcousticModelType::Linear,
            ..Default::default()
        }
    }

    /// Create configuration for Westervelt equation
    pub fn westervelt(nonlinearity_scaling: f64) -> Self {
        Self {
            model_type: AcousticModelType::Westervelt,
            nonlinearity_scaling,
            ..Default::default()
        }
    }

    /// Create configuration for Kuznetsov equation
    pub fn kuznetsov(nonlinearity_scaling: f64) -> Self {
        Self {
            model_type: AcousticModelType::Kuznetsov,
            nonlinearity_scaling,
            adaptive_time_stepping: true, // Recommended for stability
            ..Default::default()
        }
    }
}