//! Validation Module
//!
//! This module provides tools for validating the accuracy and correctness
//! of numerical solvers.

pub mod kwave_comparison;
pub mod numerical_accuracy;

pub use numerical_accuracy::{
    BoundaryResults, ConservationResults, ConvergenceResults, DispersionResults,
    NumericalValidator, StabilityResults, ValidationResults,
};

pub use kwave_comparison::{KWaveTestCase, KWaveValidator, ValidationReport};
