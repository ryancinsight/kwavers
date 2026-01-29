//! Clinical therapy workflows
//!
//! This module provides application-level therapeutic workflows that combine
//! physics models and solvers for clinical therapy applications.

pub mod domain_types;
pub mod metrics;
pub mod modalities;
pub mod parameters;
pub mod swe_3d_workflows;

// Re-export main types for convenience
pub use domain_types::{TherapyMechanism, TherapyModality, TherapyParameters, TreatmentMetrics};
pub use swe_3d_workflows::*;

// TherapyCalculator moved to crate::simulation::therapy::calculator
