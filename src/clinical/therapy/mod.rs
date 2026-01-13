//! Clinical therapy workflows
//!
//! This module provides application-level therapeutic workflows that combine
//! physics models and solvers for clinical therapy applications.

pub mod lithotripsy;
pub mod metrics;
pub mod modalities;
pub mod parameters;
pub mod swe_3d_workflows;
pub mod therapy_integration;

// Re-export main types for convenience
pub use crate::physics::acoustics::therapy::cavitation::{
    CavitationDetectionMethod, TherapyCavitationDetector,
};
pub use metrics::TreatmentMetrics;
pub use modalities::{TherapyMechanism, TherapyModality};
pub use parameters::TherapyParameters;
pub use swe_3d_workflows::*;
pub use therapy_integration::*;

pub use crate::simulation::therapy::calculator::TherapyCalculator;

// TherapyCalculator moved to crate::simulation::therapy::calculator
