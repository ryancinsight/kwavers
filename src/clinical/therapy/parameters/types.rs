//! Therapy parameters

/// Therapy treatment parameters
#[derive(Debug, Clone, Copy)]
pub struct TherapyParameters {
    /// Transmit frequency (Hz)
    pub frequency: f64,
    /// Peak negative pressure (Pa)
    pub peak_negative_pressure: f64,
    /// Treatment duration (s)
    pub treatment_duration: f64,
    /// Mechanical index (MI) - derived
    pub mechanical_index: f64,
    /// Duty cycle (0.0 - 1.0)
    pub duty_cycle: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
}

impl TherapyParameters {
    /// Create new parameters
    pub fn new(frequency: f64, pressure: f64, duration: f64) -> Self {
        let mut params = Self {
            frequency,
            peak_negative_pressure: pressure,
            treatment_duration: duration,
            mechanical_index: 0.0,
            duty_cycle: 1.0,
            prf: 0.0,
        };
        params.calculate_mechanical_index();
        params
    }

    /// Create default HIFU parameters
    pub fn hifu() -> Self {
        Self::new(1.5e6, 2.0e6, 5.0) // 1.5 MHz, 2 MPa, 5s
    }

    /// Calculate Mechanical Index (MI)
    pub fn calculate_mechanical_index(&mut self) {
        if self.frequency > 0.0 {
            // MI = PNP (MPa) / sqrt(f (MHz))
            self.mechanical_index =
                (self.peak_negative_pressure / 1e6) / (self.frequency / 1e6).sqrt();
        }
    }

    /// Validate safety parameters
    pub fn validate_safety(&self) -> bool {
        // Basic safety checks
        if self.mechanical_index > 1.9 {
            // FDA limit for diagnostic, heuristic for therapy
            // Therapy can go higher, but warning needed
            return false;
        }
        if self.treatment_duration > 3600.0 {
            return false;
        }
        true
    }
}
