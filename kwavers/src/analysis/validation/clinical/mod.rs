//! Clinical Validation Framework for Medical Ultrasound.
//!
//! SRP split:
//! - `types`     — all public data types
//! - `validator` — `ClinicalValidator` struct + `Default` + `new()`
//! - `validate`  — B-mode, Doppler, safety validation + score calculation
//! - `report`    — `generate_validation_report`

mod report;
#[cfg(test)]
mod tests;
mod types;
mod validate;
mod validator;

pub use types::{
    ClinicalCategory, ClinicalRequirements, ClinicalStandard, ClinicalValidationResult,
    DopplerValidationThresholds, ImageQualityMetrics, MeasurementAccuracy, SafetyIndices,
};
pub use validator::ClinicalValidator;
