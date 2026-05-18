//! Safety Monitoring for Transcranial Focused Ultrasound
//!
//! Real-time monitoring of thermal and mechanical indices during tFUS treatments
//! to ensure patient safety and treatment efficacy.

pub mod mechanical_index;
pub mod monitor;
pub mod safety_checks;
pub mod thermal_dose;
pub mod types;

#[cfg(test)]
mod tests;

pub use monitor::TranscranialSafetyMonitor;
pub use types::{
    MechanicalIndex, SafetyReport, SafetyThresholds, TranscranialSafetyDose,
    TranscranialSafetyLevel, TranscranialSafetyStatus, TreatmentProgress,
};
