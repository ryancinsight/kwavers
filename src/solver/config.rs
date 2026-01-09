//! Solver configuration parameters

use serde::{Deserialize, Serialize};

/// Solver configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverParameters {
    /// Solver type
    pub solver_type: SolverType,
    /// Time integration scheme
    pub time_scheme: TimeScheme,
    /// Spatial discretization order
    pub spatial_order: usize,
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use adaptive time stepping
    pub adaptive_dt: bool,
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

impl SolverParameters {
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

        if self.max_iterations == 0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "max_iterations".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.tolerance <= 0.0 || self.tolerance >= 1.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "tolerance".to_string(),
                value: self.tolerance.to_string(),
                constraint: "Must be in (0, 1)".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for SolverParameters {
    fn default() -> Self {
        Self {
            solver_type: SolverType::FDTD,
            time_scheme: TimeScheme::Leapfrog,
            spatial_order: 4,
            max_iterations: 1000,
            tolerance: 1e-6,
            adaptive_dt: false,
        }
    }
}
