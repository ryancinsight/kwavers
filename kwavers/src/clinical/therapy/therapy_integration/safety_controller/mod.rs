//! Real-Time Safety Control System for Therapy Execution.
//!
//! Implements active safety enforcement for therapeutic ultrasound,
//! enabling real-time monitoring and adaptive therapy control to prevent adverse effects.
//!
//! Enforced limits: Thermal Index (TI), Mechanical Index (MI), Cavitation Dose,
//! Treatment Time, and per-Organ Dose.

pub mod controller;
pub mod types;
#[cfg(test)]
mod tests;

pub use controller::SafetyController;
pub use types::TherapyAction;
