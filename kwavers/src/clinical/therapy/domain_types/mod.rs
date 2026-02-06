//! Therapy domain types
//!
//! Core types for therapeutic ultrasound applications.

/// Treatment metrics for therapy monitoring
#[derive(Debug, Clone, Default)]
pub struct TreatmentMetrics {
    /// Thermal dose in CEM43
    pub thermal_dose: f64,
    /// Cavitation dose
    pub cavitation_dose: f64,
    /// Peak temperature reached (°C)
    pub peak_temperature: f64,
    /// Safety index (0-1)
    pub safety_index: f64,
    /// Treatment efficiency (0-1)
    pub efficiency: f64,
}

/// Therapy mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyMechanism {
    /// Thermal mechanism
    Thermal,
    /// Mechanical mechanism
    Mechanical,
    /// Combined thermal and mechanical
    Combined,
}

/// Therapy modality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyModality {
    /// High-intensity focused ultrasound
    HIFU,
    /// Low-intensity focused ultrasound
    LIFU,
    /// Histotripsy
    Histotripsy,
}

/// Therapy parameters
#[derive(Debug, Clone, Copy, Default)]
pub struct TherapyParameters {
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

impl TherapyParameters {
    /// Create new therapy parameters
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
    pub fn hifu() -> Self {
        Self {
            frequency: 1.5e6,
            pressure: 2.0e6,
            duration: 5.0,
            peak_negative_pressure: 2.0e6,
            mechanical_index: 1.5,
            treatment_duration: 5.0,
            duty_cycle: 0.5,
            prf: 100.0,
        }
    }
}
