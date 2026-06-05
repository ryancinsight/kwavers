//! HIFU Treatment Planning System.
//!
//! Focal dimensions from O'Neil (1949): FWHM_lat = 1.02·λ·F#, FWHM_ax = (8/π)·λ·F#².

mod planner;
mod schedule;
#[cfg(test)]
mod tests;
mod types;

pub use planner::HIFUPlanner;
pub use schedule::{SonicationSchedule, SonicationSubspot};
pub use types::{
    AblationTarget, ClinicalHIFUTransducer, ClinicalHIFUTreatmentPlan, FocalSpot,
    FocalSpotDoseEstimate, TreatmentFeasibility,
};
