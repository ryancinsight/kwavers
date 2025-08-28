//! Validation Module
//!
//! This module provides tools for validating the accuracy and correctness
//! of numerical solvers.

pub mod kwave;
pub mod numerical_accuracy;

pub use numerical_accuracy::{
    BoundaryResults, ConservationResults, ConvergenceResults, DispersionResults,
    NumericalValidator, StabilityResults, ValidationResults,
};

pub use kwave::{report::ValidationReport, test_cases::KWaveTestCase, validator::KWaveValidator};
