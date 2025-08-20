//! Physics validation module - domain-based organization
//! 
//! Validates numerical implementations against analytical solutions from literature

pub mod wave;
pub mod nonlinear;
pub mod material;
pub mod numerical;
pub mod conservation;

use crate::error::KwaversResult;

/// Validation metrics for comparing numerical and analytical solutions
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub l2_error: f64,
    pub linf_error: f64,
    pub relative_error: f64,
    pub convergence_rate: Option<f64>,
}

impl ValidationMetrics {
    pub fn new(l2: f64, linf: f64, relative: f64) -> Self {
        Self {
            l2_error: l2,
            linf_error: linf,
            relative_error: relative,
            convergence_rate: None,
        }
    }
    
    pub fn with_convergence(mut self, rate: f64) -> Self {
        self.convergence_rate = Some(rate);
        self
    }
    
    pub fn passes_tolerance(&self, tol: f64) -> bool {
        self.relative_error < tol
    }
}

/// Common trait for physics validators
pub trait PhysicsValidator {
    /// Run validation and return metrics
    fn validate(&self) -> KwaversResult<ValidationMetrics>;
    
    /// Get name of validation test
    fn name(&self) -> &str;
    
    /// Get literature reference
    fn reference(&self) -> &str;
}