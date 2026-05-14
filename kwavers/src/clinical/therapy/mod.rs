//! Clinical therapy workflows
//!
//! This module provides application-level therapeutic workflows that combine
//! physics models and solvers for clinical therapy applications.

pub mod clinical_scenarios;
pub mod domain_types;
pub mod hifu_planning;
pub mod lithotripsy;
pub mod metrics;
pub mod microbubble_dynamics;
pub mod modalities;
pub mod parameters;
pub mod swe_3d_workflows;
pub mod theranostic_guidance;

pub mod therapy_integration;

// Re-export main types for convenience
pub use clinical_scenarios::{
    intrinsic_threshold_pa, BenefitDetriment, HistotripsyRegime, HistotripsyScenario, PulsePattern,
};
pub use domain_types::{TherapyMechanism, TherapyModality, TherapyParameters, TreatmentMetrics};
pub use hifu_planning::{
    AblationTarget, FocalSpot, HIFUPlanner, HIFUTransducer, HIFUTreatmentPlan, SonicationSchedule,
    SonicationSubspot, ThermalDose, TreatmentFeasibility,
};
// 3D shear wave elastography workflows
pub use swe_3d_workflows::{
    ClinicalDecisionSupport, ElasticityMap2D, ElasticityMap3D, FibrosisStage,
    MultiPlanarReconstruction, VolumetricROI, VolumetricStatistics,
};

// TherapyCalculator moved to crate::simulation::therapy::calculator
