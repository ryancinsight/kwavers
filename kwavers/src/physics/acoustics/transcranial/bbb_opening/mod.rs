//! Blood-Brain Barrier Opening with Focused Ultrasound
//!
//! Models the mechanisms of BBB opening for targeted drug delivery using
//! low-intensity focused ultrasound with microbubbles.

mod models;
mod optimization;
mod safety;
mod simulator;
mod types;

#[cfg(test)]
mod tests;

pub use simulator::BBBOpening;
pub use types::{BBBParameters, PermeabilityEnhancement, SafetyValidation, TreatmentProtocol};
