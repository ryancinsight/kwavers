//! IMEX (Implicit-Explicit) Time Integration Schemes
//! 
//! This module provides advanced time integration methods that combine
//! implicit and explicit treatments for different terms in the equations.
//! This is particularly useful for problems with multiple time scales
//! where some terms are stiff (requiring implicit treatment) and others
//! are non-stiff (allowing explicit treatment).
//!
//! # Features
//!
//! - IMEX Runge-Kutta schemes (IMEX-RK)
//! - IMEX Backward Differentiation Formulas (IMEX-BDF)
//! - Operator splitting strategies
//! - Automatic stiffness detection
//! - Adaptive time stepping with stability control
//! - Linear and nonlinear implicit solvers
//!
//! # Design Principles
//!
//! - **SOLID**: Each component has a single responsibility
//! - **DRY**: Common functionality is abstracted into traits
//! - **KISS**: Simple interfaces for complex algorithms
//! - **Clean Code**: Clear naming and documentation

pub mod traits;
pub mod imex_rk;
pub mod imex_bdf;
pub mod operator_splitting;
pub mod implicit_solver;
pub mod stability;
pub mod stiffness_detection;

#[cfg(test)]
mod tests;

use ndarray::Array3;
use std::sync::Arc;
use crate::grid::Grid;
use crate::error::KwaversResult;

// Re-export main types
pub use traits::{IMEXScheme, IMEXConfig, OperatorSplitting, StiffnessIndicator};
pub use imex_rk::{IMEXRK, IMEXRKConfig, IMEXRKType};
pub use imex_bdf::{IMEXBDF, IMEXBDFConfig};
pub use operator_splitting::{StrangSplitting, LieTrotterSplitting};
pub use implicit_solver::{ImplicitSolver, LinearSolver, NewtonSolver};
pub use stability::{IMEXStabilityAnalyzer, StabilityRegion};
pub use stiffness_detection::{StiffnessDetector, StiffnessMetric};

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
            IMEXSchemeType::RungeKutta(rk) => rk.step(field, dt, explicit_rhs, implicit_rhs, implicit_solver),
            IMEXSchemeType::BDF(bdf) => bdf.step(field, dt, explicit_rhs, implicit_rhs, implicit_solver),
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
    Newton(NewtonSolver),
}

impl ImplicitSolverType {
    fn solve<F>(
        &self,
        initial_guess: &Array3<f64>,
        residual_fn: F,
    ) -> KwaversResult<Array3<f64>>
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
    fn solve<F>(
        &self,
        initial_guess: &Array3<f64>,
        residual_fn: F,
    ) -> KwaversResult<Array3<f64>>
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
    pub fn new(
        config: IMEXConfig,
        scheme: IMEXSchemeType,
        grid: Arc<Grid>,
    ) -> Self {
        let implicit_solver = ImplicitSolverType::Newton(
            NewtonSolver::new(config.tolerance, config.max_iterations)
        );
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
            let stiffness = self.stiffness_detector.detect(field, &explicit_rhs, &implicit_rhs)?;
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
                        condition: format!("dt={:.3e} exceeds stable dt={:.3e}", dt, stable_dt),
                    }
                ));
            }
        }
        
        // Perform IMEX time step
        self.scheme.step(
            field,
            dt,
            explicit_rhs,
            implicit_rhs,
            &self.implicit_solver,
        )
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