//! Treatment Planning for Transcranial Focused Ultrasound
//!
//! Patient-specific treatment planning using CT scans for skull characterization
//! and optimal trajectory calculation for brain targets.

pub mod optimization;
pub mod planner;
pub mod safety;
pub mod simulation;
pub mod skull_analysis;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export common public types
pub use planner::TreatmentPlanner;
pub use types::{
    SafetyConstraints, SkullProperties, TargetShape, TargetVolume, TransducerSetup,
    TransducerSpecification, TreatmentPlan,
};
