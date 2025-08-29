//! Therapy parameter definitions and presets
//!
//! Provides parameter structures and presets for different therapy modalities.

use super::modalities::TherapyModality;

/// Therapy treatment parameters
#[derive(Debug, Clone)]
pub struct TherapyParameters {
    /// Acoustic frequency [Hz]
    pub frequency: f64,
    /// Peak negative pressure [Pa]
    pub peak_negative_pressure: f64,
    /// Peak positive pressure [Pa]
    pub peak_positive_pressure: f64,
    /// Pulse duration [s]
    pub pulse_duration: f64,
    /// Pulse repetition frequency [Hz]
    pub prf: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
    /// Total treatment duration [s]
    pub treatment_duration: f64,
    /// Mechanical index (MI)
    pub mechanical_index: f64,
    /// Thermal index (TI)
    pub thermal_index: f64,
}

impl Default for TherapyParameters {
    fn default() -> Self {
        Self {
            frequency: 1e6,
            peak_negative_pressure: 1e6,
            peak_positive_pressure: 1e6,
            pulse_duration: 1e-3,
            prf: 100.0,
            duty_cycle: 0.1,
            treatment_duration: 60.0,
            mechanical_index: 0.0,
            thermal_index: 0.0,
        }
    }
}

impl TherapyParameters {
    /// Create parameters for HIFU therapy
    pub fn hifu() -> Self {
        Self {
            frequency: 1.5e6,             // 1.5 MHz
            peak_negative_pressure: 3e6,  // 3 MPa
            peak_positive_pressure: 10e6, // 10 MPa
            pulse_duration: 0.1,          // 100 ms continuous wave
            prf: 10.0,                    // 10 Hz
            duty_cycle: 1.0,              // 100% duty cycle (CW)
            treatment_duration: 10.0,     // 10 seconds
            mechanical_index: 0.0,
            thermal_index: 0.0,
        }
    }

    /// Create parameters for LIFU neuromodulation
    pub fn lifu() -> Self {
        Self {
            frequency: 0.5e6,              // 500 kHz
            peak_negative_pressure: 0.3e6, // 0.3 MPa
            peak_positive_pressure: 0.3e6, // 0.3 MPa
            pulse_duration: 0.5,           // 500 ms
            prf: 1.0,                      // 1 Hz
            duty_cycle: 0.5,               // 50% duty cycle
            treatment_duration: 300.0,     // 5 minutes
            mechanical_index: 0.0,
            thermal_index: 0.0,
        }
    }

    /// Create parameters for histotripsy
    pub fn histotripsy() -> Self {
        Self {
            frequency: 1e6,               // 1 MHz
            peak_negative_pressure: 30e6, // 30 MPa (very high)
            peak_positive_pressure: 80e6, // 80 MPa
            pulse_duration: 10e-6,        // 10 μs pulses
            prf: 1000.0,                  // 1 kHz PRF
            duty_cycle: 0.01,             // 1% duty cycle
            treatment_duration: 60.0,     // 1 minute
            mechanical_index: 0.0,
            thermal_index: 0.0,
        }
    }

    /// Create parameters for BBB opening
    pub fn bbb_opening() -> Self {
        Self {
            frequency: 0.25e6,             // 250 kHz
            peak_negative_pressure: 0.3e6, // 0.3 MPa (with microbubbles)
            peak_positive_pressure: 0.3e6, // 0.3 MPa
            pulse_duration: 10e-3,         // 10 ms bursts
            prf: 1.0,                      // 1 Hz
            duty_cycle: 0.01,              // 1% duty cycle
            treatment_duration: 120.0,     // 2 minutes
            mechanical_index: 0.6,         // Safe with microbubbles
            thermal_index: 0.3,
        }
    }

    /// Create parameters from modality
    pub fn from_modality(modality: TherapyModality) -> Self {
        match modality {
            TherapyModality::HIFU => Self::hifu(),
            TherapyModality::LIFU => Self::lifu(),
            TherapyModality::Histotripsy => Self::histotripsy(),
            TherapyModality::BBBOpening => Self::bbb_opening(),
            _ => Self::default(),
        }
    }

    /// Calculate mechanical index: MI = P_neg / sqrt(f)
    pub fn calculate_mechanical_index(&mut self) {
        self.mechanical_index = self.peak_negative_pressure / (self.frequency.sqrt() * 1e6);
    }

    /// Calculate thermal index (simplified)
    pub fn calculate_thermal_index(&mut self, intensity: f64) {
        // TI = Power / Power_ref (simplified)
        const POWER_REF: f64 = 1.0; // 1 W reference
        self.thermal_index = intensity * self.duty_cycle / POWER_REF;
    }

    /// Validate safety parameters
    pub fn validate_safety(&self) -> bool {
        // FDA guidelines: MI < 1.9, TI < 6.0
        const MAX_MI: f64 = 1.9;
        const MAX_TI: f64 = 6.0;

        self.mechanical_index < MAX_MI && self.thermal_index < MAX_TI
    }
}
