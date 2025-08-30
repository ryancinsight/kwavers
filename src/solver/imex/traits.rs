//! Common traits for IMEX schemes

use super::ImplicitSolverType;
use crate::error::KwaversResult;
use ndarray::Array3;
use std::fmt::Debug;

/// Configuration for IMEX schemes
#[derive(Debug, Clone))]
pub struct IMEXConfig {
    /// Whether to use adaptive stiffness detection
    pub adaptive_stiffness: bool,
    /// Threshold for stiffness detection
    pub stiffness_threshold: f64,
    /// Whether to check stability
    pub check_stability: bool,
    /// Safety factor for stability
    pub safety_factor: f64,
    /// Maximum iterations for implicit solver
    pub max_iterations: usize,
    /// Tolerance for implicit solver
    pub tolerance: f64,
}

impl Default for IMEXConfig {
    fn default() -> Self {
        Self {
            adaptive_stiffness: true,
            stiffness_threshold: 10.0,
            check_stability: true,
            safety_factor: 0.9,
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }
}

/// Trait for IMEX time integration schemes
pub trait IMEXScheme: Debug + Send + Sync {
    /// Perform one time step
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
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>;

    /// Get the order of accuracy
    fn order(&self) -> usize;

    /// Get the number of stages
    fn stages(&self) -> usize;

    /// Check if the scheme is A-stable
    fn is_a_stable(&self) -> bool;

    /// Check if the scheme is L-stable
    fn is_l_stable(&self) -> bool;

    /// Adjust parameters for stiff problems
    fn adjust_for_stiffness(&mut self, stiffness_ratio: f64);

    /// Get stability function
    fn stability_function(&self, z: f64) -> f64;
}

/// Trait for operator splitting strategies
pub trait OperatorSplitting: Debug + Send + Sync {
    /// Split and advance the solution
    fn split_step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        operator_a: F,
        operator_b: G,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>, f64) -> KwaversResult<Array3<f64>>;

    /// Get the order of the splitting
    fn order(&self) -> usize;

    /// Get the name of the splitting method
    fn name(&self) -> &str;
}

/// Trait for stiffness indicators
pub trait StiffnessIndicator: Debug + Send + Sync {
    /// Compute stiffness indicator
    fn compute<F, G>(
        &self,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<f64>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>;

    /// Check if the problem is stiff based on the indicator
    fn is_stiff(&self, indicator: f64) -> bool;
}
