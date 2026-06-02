//! Theorem Validation and Quantitative Error Bounds
//!
//! Provides systematic validation of mathematical theorems implemented in Kwavers
//! with quantitative error bounds and convergence proofs.

mod suite;
#[cfg(test)]
mod tests;
mod validators;

/// Theorem validation results
#[derive(Debug, Clone)]
pub struct TheoremValidation {
    pub theorem: String,
    pub passed: bool,
    pub error_bound: f64,
    pub measured_error: f64,
    pub confidence: f64,
    pub details: String,
}

/// Comprehensive theorem validator
#[derive(Debug)]
pub struct TheoremValidator;
