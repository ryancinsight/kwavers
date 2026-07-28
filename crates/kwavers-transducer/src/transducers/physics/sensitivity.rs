//! Transducer Sensitivity Module
//!
//! Models transmit and receive sensitivity characteristics.

use aequitas::systems::si::{
    quantities::{
        AcousticImpedance, Area, Dimensionless, ElectricPotential, ElectricPotentialPerPressure,
        Frequency, Length, Pressure, PressurePerElectricPotential,
    },
    units::Megapascal,
};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// Transducer sensitivity parameters
///
/// Based on IEC 61102 standard for ultrasound transducers
#[derive(Debug, Clone)]
pub struct TransducerSensitivity {
    /// Transmit sensitivity (Pa/V at 1m)
    pub transmit_sensitivity: PressurePerElectricPotential,
    /// Receive sensitivity (V/Pa)
    pub receive_sensitivity: ElectricPotentialPerPressure,
    /// Round-trip sensitivity (V/V)
    pub round_trip_sensitivity: Dimensionless,
    /// Conversion efficiency (%)
    pub efficiency: f64,
    /// Maximum acoustic pressure (`MPa`)
    pub max_pressure: Pressure,
}

impl TransducerSensitivity {
    /// Calculate sensitivity from transducer parameters
    #[must_use]
    pub fn from_parameters(
        coupling: Dimensionless,
        area: Area,
        impedance: AcousticImpedance,
        _frequency: Frequency,
    ) -> Self {
        // Transmit sensitivity: pressure per volt at 1 meter
        // S_t = k * sqrt(2 * Z * P_elec / A) / r
        let electrical_power = 1.0; // 1W reference
        let distance = 1.0; // 1m reference

        let transmit_sensitivity = PressurePerElectricPotential::from_base(
            coupling.into_base()
                * (2.0 * impedance.into_base() * electrical_power / area.into_base()).sqrt()
                / distance,
        );

        // Receive sensitivity: voltage per pascal
        // S_r = k * A / (Z * c)
        let sound_speed = SOUND_SPEED_TISSUE;
        let receive_sensitivity = ElectricPotentialPerPressure::from_base(
            coupling.into_base() * area.into_base() / (impedance.into_base() * sound_speed),
        );

        // Round-trip sensitivity
        let round_trip_sensitivity = Dimensionless::from_base(
            transmit_sensitivity.into_base() * receive_sensitivity.into_base(),
        );

        // Electromechanical efficiency: η = k²ₘ × 100%
        // Where kₘ is electromechanical coupling coefficient
        // Per IEEE Std 176: "Standard on Piezoelectricity"
        let efficiency = coupling.into_base().powi(2) * 100.0;

        // Maximum pressure (typical limit for medical transducers)
        let max_pressure = Pressure::from_unit::<Megapascal>(10.0);

        Self {
            transmit_sensitivity,
            receive_sensitivity,
            round_trip_sensitivity,
            efficiency,
            max_pressure,
        }
    }

    /// Calculate pressure at a given distance and voltage
    #[must_use]
    pub fn pressure_at_distance(&self, voltage: ElectricPotential, distance: Length) -> Pressure {
        Pressure::from_base(
            self.transmit_sensitivity.into_base() * voltage.into_base() / distance.into_base(),
        )
    }

    /// Calculate received voltage for given pressure
    #[must_use]
    pub fn voltage_from_pressure(&self, pressure: Pressure) -> ElectricPotential {
        ElectricPotential::from_base(self.receive_sensitivity.into_base() * pressure.into_base())
    }

    /// Calculate SNR for given target
    ///
    /// # Arguments
    /// * `target_distance` - Distance to target (m)
    /// * `reflection_coeff` - Target reflection coefficient
    /// * `attenuation` - Tissue attenuation (dB/cm/MHz)
    /// * `frequency` - Operating frequency (Hz)
    #[must_use]
    pub fn calculate_snr(
        &self,
        target_distance: Length,
        reflection_coeff: f64,
        attenuation: f64,
        frequency: Frequency,
    ) -> f64 {
        // Two-way attenuation
        let freq_mhz = frequency.into_base() / MHZ_TO_HZ;
        let distance_m = target_distance.into_base();
        let distance_cm = distance_m * 100.0;
        let total_attenuation_db = 2.0 * attenuation * distance_cm * freq_mhz;
        let attenuation_factor = 10.0_f64.powf(-total_attenuation_db / 20.0);

        // Geometric spreading (1/r² for round trip)
        let geometric_factor = 1.0 / (distance_m * distance_m);

        // Signal level
        let signal = self.round_trip_sensitivity.into_base()
            * reflection_coeff
            * attenuation_factor
            * geometric_factor;

        // Noise level (thermal noise model)
        let noise = 1e-6; // Typical noise floor in V

        20.0 * (signal / noise).log10()
    }

    /// Check if sensitivity meets requirements
    #[must_use]
    pub fn validate_sensitivity(&self, min_snr_db: f64) -> bool {
        // Check at typical imaging depth (10 cm)
        let typical_snr = self.calculate_snr(
            Length::from_base(0.1),    // 10 cm
            0.01,                      // 1% reflection
            0.5,                       // 0.5 dB/cm/MHz
            Frequency::from_base(3e6), // 3 MHz
        );

        typical_snr >= min_snr_db
    }
}
