//! Validation Module
//!
//! This module provides tools for validating the accuracy and correctness
//! of numerical solvers.

pub mod numerical_accuracy;
pub mod kwave_comparison;

pub use numerical_accuracy::{
    NumericalValidator,
    ValidationResults,
    DispersionResults,
    StabilityResults,
    BoundaryResults,
    ConservationResults,
    ConvergenceResults,
};

pub use kwave_comparison::{
    KWaveValidator,
    KWaveTestCase,
    ValidationReport,
};