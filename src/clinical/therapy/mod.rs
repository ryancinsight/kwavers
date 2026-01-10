//! Clinical therapy workflows
//!
//! This module provides application-level therapeutic workflows that combine
//! physics models and solvers for clinical therapy applications.

// pub mod cavitation; // Moved to physics
pub mod lithotripsy;
// pub mod metrics; // Use domain
// pub mod modalities; // Use domain
// pub mod parameters; // Use domain
pub mod swe_3d_workflows;
pub mod therapy_integration;

// Re-export main types for convenience
pub use crate::domain::therapy::metrics::TreatmentMetrics;
pub use crate::domain::therapy::modalities::{TherapyMechanism, TherapyModality};
pub use crate::domain::therapy::parameters::TherapyParameters;
pub use crate::physics::acoustics::therapy::cavitation::{
    CavitationDetectionMethod, TherapyCavitationDetector,
};
pub use swe_3d_workflows::*;
pub use therapy_integration::*;

pub use crate::simulation::therapy::calculator::TherapyCalculator;
use crate::{
    domain::core::error::KwaversResult, domain::grid::Grid, domain::medium::Medium,
    physics::thermal::PennesSolver,
};
use ndarray::{Array3, Zip};
use std::sync::Arc;

// TherapyCalculator moved to crate::simulation::therapy::calculator
