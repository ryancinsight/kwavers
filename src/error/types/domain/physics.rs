//! Physics-specific error types
//!
//! Specialized error handling for physics simulations

use thiserror::Error;

/// Physics simulation errors
#[derive(Error, Debug, Clone)]
pub enum PhysicsErrorType {
    #[error("Numerical instability detected: {reason}")]
    NumericalInstability { reason: String },
    
    #[error("Physics parameter out of range: {parameter} = {value}, expected {constraint}")]
    ParameterOutOfRange {
        parameter: String,
        value: String,
        constraint: String,
    },
    
    #[error("Model incompatibility: {model1} cannot be combined with {model2}")]
    ModelIncompatibility { model1: String, model2: String },
    
    #[error("Convergence failure: {algorithm} failed to converge after {iterations} iterations")]
    ConvergenceFailure {
        algorithm: String,
        iterations: usize,
    },
    
    #[error("Boundary condition violation: {boundary_type} incompatible with {physics_model}")]
    BoundaryConditionViolation {
        boundary_type: String,
        physics_model: String,
    },
}