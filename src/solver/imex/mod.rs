//! IMEX (Implicit-Explicit) Time Integration Schemes
//!
//! This module provides time integration methods that combine
//! implicit and explicit treatments for different terms in the equations.
//! This is particularly useful for problems with multiple time scales
//! where some terms are stiff (requiring implicit treatment) and others
//! are non-stiff (allowing explicit treatment).
//!
//! ## Literature References
//!
//! 1. **Ascher, U. M., Ruuth, S. J., & Spiteri, R. J. (1997)**. "Implicit-explicit
//!    Runge-Kutta methods for time-dependent partial differential equations."
//!    *Applied Numerical Mathematics*, 25(2-3), 151-167.
//!    DOI: 10.1016/S0168-9274(97)00056-1
//!    - Original IMEX-RK formulation
//!    - Stability analysis
//!
//! 2. **Kennedy, C. A., & Carpenter, M. H. (2003)**. "Additive Runge-Kutta schemes
//!    for convection-diffusion-reaction equations." *Applied Numerical Mathematics*,
//!    44(1-2), 139-181. DOI: 10.1016/S0168-9274(02)00138-1
//!    - High-order IMEX-RK schemes
//!    - ARK methods
//!
//! 3. **Pareschi, L., & Russo, G. (2005)**. "Implicit-explicit Runge-Kutta schemes
//!    and applications to hyperbolic systems with relaxation." *Journal of Scientific
//!    Computing*, 25(1), 129-155. DOI: 10.1007/s10915-004-4636-4
//!    - IMEX for hyperbolic systems
//!    - Asymptotic preserving properties
//!
//! 4. **Frank, J., Hundsdorfer, W., & Verwer, J. G. (1997)**. "On the stability of
//!    implicit-explicit linear multistep methods." *Applied Numerical Mathematics*,
//!    25(2-3), 193-205. DOI: 10.1016/S0168-9274(97)00059-7
//!    - IMEX-BDF methods
//!    - Stability regions
//!
//! 5. **Knoth, O., & Wolke, R. (1998)**. "Implicit-explicit Runge-Kutta methods for
//!    computing atmospheric reactive flows." *Applied Numerical Mathematics*,
//!    28(2-4), 327-341. DOI: 10.1016/S0168-9274(98)00051-8
//!    - Application to atmospheric modeling
//!    - Stiffness detection strategies
//!
//! ## Features
//!
//! - IMEX Runge-Kutta schemes (IMEX-RK)
//! - IMEX Backward Differentiation Formulas (IMEX-BDF)
//! - Operator splitting strategies
//! - Automatic stiffness detection
//! - Adaptive time stepping with stability control
//! - Linear and nonlinear implicit solvers
//!
//! ## Design Principles
//!
//! - **SOLID**: Each component has a single responsibility
//! - **DRY**: Common functionality is abstracted into traits
//! - **KISS**: Simple interfaces for complex algorithms
//! - **Clean Code**: Clear naming and documentation

pub mod imex_bdf;
pub mod imex_rk;
pub mod implicit_solver;
pub mod operator_splitting;
pub mod stability;
pub mod stiffness_detection;
pub mod traits;

#[cfg(test)]
mod tests;

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;
use std::sync::Arc;

// Re-export main types
pub use imex_bdf::{IMEXBDFConfig, IMEXBDF};
pub use imex_rk::{IMEXRKConfig, IMEXRKType, IMEXRK};
pub use implicit_solver::{ImplicitSolver, LinearSolver, NonlinearSolver};
pub use operator_splitting::{LieTrotterSplitting, StrangSplitting};
pub use stability::{IMEXStabilityAnalyzer, StabilityRegion};
pub use stiffness_detection::{StiffnessDetector, StiffnessMetric};
pub use traits::{IMEXConfig, IMEXScheme, OperatorSplitting, StiffnessIndicator};

/// Enum wrapper for IMEX schemes
#[derive(Debug)]
pub enum IMEXSchemeType {
    RungeKutta(IMEXRK),
    BDF(IMEXBDF),
}

impl IMEXSchemeType {
    fn step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        explicit_rhs: F,
        implicit_rhs: G,
        implicit_solver: &ImplicitSolverType,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        match self {
            IMEXSchemeType::RungeKutta(rk) => {
                rk.step(field, dt, explicit_rhs, implicit_rhs, implicit_solver)
            }
            IMEXSchemeType::BDF(bdf) => {
                bdf.step(field, dt, explicit_rhs, implicit_rhs, implicit_solver)
            }
        }
    }

    fn order(&self) -> usize {
        match self {
            IMEXSchemeType::RungeKutta(rk) => rk.order(),
            IMEXSchemeType::BDF(bdf) => bdf.order(),
        }
    }

    fn stages(&self) -> usize {
        match self {
            IMEXSchemeType::RungeKutta(rk) => rk.stages(),
            IMEXSchemeType::BDF(bdf) => bdf.stages(),
        }
    }

    fn is_a_stable(&self) -> bool {
        match self {
            IMEXSchemeType::RungeKutta(rk) => rk.is_a_stable(),
            IMEXSchemeType::BDF(bdf) => bdf.is_a_stable(),
        }
    }

    fn is_l_stable(&self) -> bool {
        match self {
            IMEXSchemeType::RungeKutta(rk) => rk.is_l_stable(),
            IMEXSchemeType::BDF(bdf) => bdf.is_l_stable(),
        }
    }

    fn adjust_for_stiffness(&mut self, stiffness_ratio: f64) {
        match self {
            IMEXSchemeType::RungeKutta(rk) => rk.adjust_for_stiffness(stiffness_ratio),
            IMEXSchemeType::BDF(bdf) => bdf.adjust_for_stiffness(stiffness_ratio),
        }
    }

    fn stability_function(&self, z: f64) -> f64 {
        match self {
            IMEXSchemeType::RungeKutta(rk) => rk.stability_function(z),
            IMEXSchemeType::BDF(bdf) => bdf.stability_function(z),
        }
    }
}

/// Enum wrapper for implicit solvers
#[derive(Debug)]
pub enum ImplicitSolverType {
    Linear(LinearSolver),
    Newton(NonlinearSolver),
}

impl ImplicitSolverType {
    fn solve<F>(&self, initial_guess: &Array3<f64>, residual_fn: F) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        match self {
            ImplicitSolverType::Linear(solver) => solver.solve(initial_guess, residual_fn),
            ImplicitSolverType::Newton(solver) => solver.solve(initial_guess, residual_fn),
        }
    }
}

impl ImplicitSolver for ImplicitSolverType {
    fn solve<F>(&self, initial_guess: &Array3<f64>, residual_fn: F) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        match self {
            ImplicitSolverType::Linear(solver) => solver.solve(initial_guess, residual_fn),
            ImplicitSolverType::Newton(solver) => solver.solve(initial_guess, residual_fn),
        }
    }

    fn tolerance(&self) -> f64 {
        match self {
            ImplicitSolverType::Linear(solver) => solver.tolerance(),
            ImplicitSolverType::Newton(solver) => solver.tolerance(),
        }
    }

    fn max_iterations(&self) -> usize {
        match self {
            ImplicitSolverType::Linear(solver) => solver.max_iterations(),
            ImplicitSolverType::Newton(solver) => solver.max_iterations(),
        }
    }
}

/// Main IMEX time integrator
#[derive(Debug)]
pub struct IMEXIntegrator {
    /// Configuration
    config: IMEXConfig,
    /// IMEX scheme
    scheme: IMEXSchemeType,
    /// Implicit solver
    implicit_solver: ImplicitSolverType,
    /// Stiffness detector
    stiffness_detector: StiffnessDetector,
    /// Stability analyzer
    stability_analyzer: IMEXStabilityAnalyzer,
    /// Grid reference
    grid: Arc<Grid>,
}

impl IMEXIntegrator {
    /// Create a new IMEX integrator
    pub fn new(config: IMEXConfig, scheme: IMEXSchemeType, grid: Arc<Grid>) -> Self {
        let implicit_solver = ImplicitSolverType::Newton(NonlinearSolver::new(
            config.tolerance,
            config.max_iterations,
        ));
        let stiffness_detector = StiffnessDetector::new(config.stiffness_threshold);
        let stability_analyzer = IMEXStabilityAnalyzer::new();

        Self {
            config,
            scheme,
            implicit_solver,
            stiffness_detector,
            stability_analyzer,
            grid,
        }
    }

    /// Advance solution by one time step
    pub fn advance<F, G>(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        explicit_rhs: F,
        implicit_rhs: G,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Check stiffness if adaptive
        if self.config.adaptive_stiffness {
            let stiffness = self
                .stiffness_detector
                .detect(field, &explicit_rhs, &implicit_rhs)?;
            if stiffness.is_stiff() {
                // Adjust scheme parameters for stiff problems
                self.scheme.adjust_for_stiffness(stiffness.ratio());
            }
        }

        // Check stability
        if self.config.check_stability {
            let stable_dt = self.stability_analyzer.max_stable_timestep(
                &self.scheme,
                field,
                &explicit_rhs,
                &implicit_rhs,
            )?;

            if dt > stable_dt * self.config.safety_factor {
                return Err(crate::error::KwaversError::Numerical(
                    crate::error::NumericalError::Instability {
                        operation: "IMEX time step".to_string(),
                        condition: dt,
                    },
                ));
            }
        }

        // Perform IMEX time step with error context
        self.scheme
            .step(field, dt, explicit_rhs, implicit_rhs, &self.implicit_solver)
            .map_err(|e| {
                // Add context about the current scheme and conditions
                let scheme_name = match &self.scheme {
                    IMEXSchemeType::RungeKutta(rk) => format!("IMEX-RK (order {})", rk.order()),
                    IMEXSchemeType::BDF(bdf) => format!("IMEX-BDF (order {})", bdf.order()),
                };

                crate::error::KwaversError::Numerical(
                    crate::error::NumericalError::InvalidOperation(format!(
                        "IMEX scheme {} failed at dt={:.3e}: {}",
                        scheme_name, dt, e
                    )),
                )
            })
    }

    /// Get current stiffness metric
    pub fn stiffness_metric(&self) -> Option<StiffnessMetric> {
        self.stiffness_detector.last_metric()
    }

    /// Get stability region
    pub fn stability_region(&self) -> StabilityRegion {
        self.stability_analyzer.compute_region(&self.scheme)
    }
}
