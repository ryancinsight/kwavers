//! Clinical therapy workflows
//!
//! This module provides application-level therapeutic workflows that combine
//! physics models and solvers for clinical therapy applications.

pub mod domain_types;
pub mod hifu_planning;
pub mod lithotripsy;
pub mod metrics;
pub mod modalities;
pub mod parameters;
pub mod swe_3d_workflows;

// Note: therapy_integration module exists but not publicly exposed yet
// due to ongoing integration work (Sprint 214 Session 5)
// pub mod therapy_integration;

// Re-export main types for convenience
pub use domain_types::{TherapyMechanism, TherapyModality, TherapyParameters, TreatmentMetrics};
pub use hifu_planning::{
    AblationTarget, FocalSpot, HIFUPlanner, HIFUTransducer, HIFUTreatmentPlan, ThermalDose,
    TreatmentFeasibility,
};
// 3D shear wave elastography workflows
pub use swe_3d_workflows::{
    ClinicalDecisionSupport, ElasticityMap2D, ElasticityMap3D, FibrosisStage,
    MultiPlanarReconstruction, VolumetricROI, VolumetricStatistics,
};

// TherapyCalculator moved to crate::simulation::therapy::calculator
