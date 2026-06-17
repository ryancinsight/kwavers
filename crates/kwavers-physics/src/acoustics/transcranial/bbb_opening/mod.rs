//! Blood-Brain Barrier Opening with Focused Ultrasound
//!
//! Models the mechanisms of BBB opening for targeted drug delivery using
//! low-intensity focused ultrasound with microbubbles.

pub mod dose_response;
mod models;
mod optimization;
mod safety;
mod simulator;
mod types;

#[cfg(test)]
mod tests;

pub use dose_response::{bbb_acoustic_dose, bbb_closure_permeability, bbb_permeability_hill};
pub use simulator::BBBOpening;
pub use types::{BBBParameters, BbbTreatmentProtocol, PermeabilityEnhancement, SafetyValidation};
