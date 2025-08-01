//! Validation module for numerical methods
//! 
//! This module provides comprehensive validation and testing infrastructure
//! for all numerical solvers in the kwavers framework.

pub mod numerical_accuracy;

pub use numerical_accuracy::{
    NumericalValidator, ValidationResults, DispersionResults,
    StabilityResults, BoundaryResults, ConservationResults,
    ConvergenceResults, report_validation_results
};