//! Real-Time Safety Control System for Therapy Execution.
//!
//! Implements active safety enforcement for therapeutic ultrasound,
//! enabling real-time monitoring and adaptive therapy control to prevent adverse effects.
//!
//! Enforced limits: Thermal Index (TI), Mechanical Index (MI), Cavitation Dose,
//! Treatment Time, and per-Organ Dose.

pub mod controller;
#[cfg(test)]
mod tests;
pub mod types;

pub use controller::SafetyController;
pub use types::TherapyAction;
