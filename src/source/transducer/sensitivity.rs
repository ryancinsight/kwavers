//! Transducer Sensitivity Module
//!
//! Models transmit and receive sensitivity characteristics.

/// Transducer sensitivity parameters
///
/// Based on IEC 61102 standard for ultrasound transducers
#[derive(Debug, Clone)]
pub struct TransducerSensitivity {
    /// Transmit sensitivity (Pa/V at 1m)
    pub transmit_sensitivity: f64,
    /// Receive sensitivity (V/Pa)
    pub receive_sensitivity: f64,
    /// Round-trip sensitivity (V/V)
    pub round_trip_sensitivity: f64,
    /// Conversion efficiency (%)
    pub efficiency: f64,
    /// Maximum acoustic pressure (MPa)
    pub max_pressure: f64,
}

impl TransducerSensitivity {
    /// Calculate sensitivity from transducer parameters
    pub fn from_parameters(coupling: f64, area: f64, impedance: f64, frequency: f64) -> Self {
        // Transmit sensitivity: pressure per volt at 1 meter
        // S_t = k * sqrt(2 * Z * P_elec / A) / r
        let electrical_power = 1.0; // 1W reference
        let distance = 1.0; // 1m reference

        let transmit_sensitivity =
            coupling * (2.0 * impedance * electrical_power / area).sqrt() / distance;

        // Receive sensitivity: voltage per pascal
        // S_r = k * A / (Z * c)
        let sound_speed = 1540.0;
        let receive_sensitivity = coupling * area / (impedance * sound_speed);

        // Round-trip sensitivity
        let round_trip_sensitivity = transmit_sensitivity * receive_sensitivity;

        // Efficiency (simplified model)
        let efficiency = coupling.powi(2) * 100.0;

        // Maximum pressure (typical limit for medical transducers)
        let max_pressure = 10.0; // MPa

        Self {
            transmit_sensitivity,
            receive_sensitivity,
            round_trip_sensitivity,
            efficiency,
            max_pressure,
        }
    }

    /// Calculate pressure at a given distance and voltage
    pub fn pressure_at_distance(&self, voltage: f64, distance: f64) -> f64 {
        self.transmit_sensitivity * voltage / distance
    }

    /// Calculate received voltage for given pressure
    pub fn voltage_from_pressure(&self, pressure: f64) -> f64 {
        self.receive_sensitivity * pressure
    }

    /// Calculate SNR for given target
    ///
    /// # Arguments
    /// * `target_distance` - Distance to target (m)
    /// * `reflection_coeff` - Target reflection coefficient
    /// * `attenuation` - Tissue attenuation (dB/cm/MHz)
    /// * `frequency` - Operating frequency (Hz)
    pub fn calculate_snr(
        &self,
        target_distance: f64,
        reflection_coeff: f64,
        attenuation: f64,
        frequency: f64,
    ) -> f64 {
        // Two-way attenuation
        let freq_mhz = frequency / 1e6;
        let distance_cm = target_distance * 100.0;
        let total_attenuation_db = 2.0 * attenuation * distance_cm * freq_mhz;
        let attenuation_factor = 10.0_f64.powf(-total_attenuation_db / 20.0);

        // Geometric spreading (1/r² for round trip)
        let geometric_factor = 1.0 / (target_distance * target_distance);

        // Signal level
        let signal =
            self.round_trip_sensitivity * reflection_coeff * attenuation_factor * geometric_factor;

        // Noise level (thermal noise model)
        let noise = 1e-6; // Typical noise floor in V

        20.0 * (signal / noise).log10()
    }

    /// Check if sensitivity meets requirements
    pub fn validate_sensitivity(&self, min_snr_db: f64) -> bool {
        // Check at typical imaging depth (10 cm)
        let typical_snr = self.calculate_snr(
            0.1,  // 10 cm
            0.01, // 1% reflection
            0.5,  // 0.5 dB/cm/MHz
            3e6,  // 3 MHz
        );

        typical_snr >= min_snr_db
    }
}
