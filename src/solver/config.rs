//! Solver configuration parameters
//!
//! Consolidated configuration for all solver types.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Unified solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfiguration {
    /// Solver type
    pub solver_type: SolverType,
    /// Time integration scheme
    pub time_scheme: TimeScheme,
    /// Spatial discretization order
    pub spatial_order: usize,
    /// Maximum number of time steps (was max_steps in interface config)
    pub max_steps: usize,
    /// Time step size (was dt in interface config)
    pub dt: f64,
    /// CFL number (was cfl in interface config)
    pub cfl: f64,
    /// Convergence tolerance (was tolerance)
    pub tolerance: f64,
    /// Maximum iterations (was max_iterations)
    pub max_iterations: usize,
    /// Use adaptive time stepping
    pub adaptive_dt: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Enable adaptive mesh refinement
    pub enable_amr: bool,
    /// Progress reporting interval
    pub progress_interval: Duration,
    /// Validation mode
    pub validation_mode: bool,
    /// Detailed logging
    pub detailed_logging: bool,
}

/// Types of wave equation solvers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SolverType {
    /// Finite Difference Time Domain
    FDTD,
    /// Pseudo-spectral Time Domain
    PSTD,
    /// k-space pseudo-spectral
    KSpace,
    /// Discontinuous Galerkin
    DiscontinuousGalerkin,
    /// Finite Element Method
    FEM,
    /// Automatically selected
    Auto,
}

/// Time integration schemes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TimeScheme {
    /// Forward Euler (1st order)
    ForwardEuler,
    /// Leapfrog (2nd order)
    Leapfrog,
    /// Runge-Kutta 4 (4th order)
    RungeKutta4,
    /// Adams-Bashforth (multi-step)
    AdamsBashforth,
}

impl Default for SolverConfiguration {
    fn default() -> Self {
        Self {
            solver_type: SolverType::FDTD,
            time_scheme: TimeScheme::Leapfrog,
            spatial_order: 4,
            max_steps: 1000,
            dt: 1e-7,
            cfl: 0.3,
            tolerance: 1e-6,
            max_iterations: 1000,
            adaptive_dt: false,
            enable_gpu: false,
            enable_amr: false,
            progress_interval: Duration::from_secs(10),
            validation_mode: false,
            detailed_logging: false,
        }
    }
}

impl SolverConfiguration {
    /// Validate solver parameters
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if self.spatial_order == 0 || self.spatial_order > 16 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "spatial_order".to_string(),
                value: self.spatial_order.to_string(),
                constraint: "Must be between 1 and 16".to_string(),
            }
            .into());
        }

        if self.max_steps == 0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "max_steps".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.dt <= 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: self.dt.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "cfl".to_string(),
                value: self.cfl.to_string(),
                constraint: "Must be between 0 and 1".to_string(),
            }
            .into());
        }

        Ok(())
    }

     /// Create a configuration optimized for accuracy
    pub fn accuracy_optimized() -> Self {
        Self {
            cfl: 0.1,
            spatial_order: 6,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            cfl: 0.5,
            spatial_order: 2,
            enable_gpu: true,
            progress_interval: Duration::from_secs(30),
            ..Default::default()
        }
    }
}
