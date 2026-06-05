//! Therapy application parameter type.
//!
//! The application-level acoustic exposure parameters used by the therapy
//! planning and safety workflows. The canonical domain value types
//! (`DomainTreatmentMetrics`, `DomainTherapyMechanism`, `DomainTherapyModality`)
//! live in `kwavers_physics::therapy::types` (SSOT) — workflows import those
//! directly. This module holds only `ClinicalTherapyParameters`, which extends
//! the domain parameter set with planning-convenience fields and constructors.

use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

/// Therapy parameters
#[derive(Debug, Clone, Copy, Default)]
pub struct ClinicalTherapyParameters {
    /// Frequency (Hz)
    pub frequency: f64,
    /// Pressure (Pa)
    pub pressure: f64,
    /// Duration (s)
    pub duration: f64,
    /// Peak negative pressure (Pa)
    pub peak_negative_pressure: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Treatment duration (s)
    pub treatment_duration: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
}

impl ClinicalTherapyParameters {
    /// Create new therapy parameters
    #[must_use]
    pub fn new(frequency: f64, pressure: f64, duration: f64) -> Self {
        Self {
            frequency,
            pressure,
            duration,
            peak_negative_pressure: pressure,
            mechanical_index: 0.0,
            treatment_duration: duration,
            duty_cycle: 0.5,
            prf: 100.0,
        }
    }

    /// HIFU standard parameters
    #[must_use]
    pub fn hifu() -> Self {
        Self {
            frequency: 1.5 * MHZ_TO_HZ,
            pressure: 2.0 * MPA_TO_PA,
            duration: 5.0,
            peak_negative_pressure: 2.0 * MPA_TO_PA,
            mechanical_index: 1.5,
            treatment_duration: 5.0,
            duty_cycle: 0.5,
            prf: 100.0,
        }
    }
}
