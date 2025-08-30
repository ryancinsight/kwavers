//! Stokes absorption model for viscous fluids

use serde::{Deserialize, Serialize};

/// Stokes absorption parameters for viscous fluids
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StokesParameters {
    /// Dynamic viscosity [Pa·s]
    pub viscosity: f64,
    /// Bulk viscosity [Pa·s]
    pub bulk_viscosity: f64,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,
    /// Specific heat at constant pressure [J/(kg·K)]
    pub specific_heat_p: f64,
    /// Specific heat at constant volume [J/(kg·K)]
    pub specific_heat_v: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
}

impl Default for StokesParameters {
    fn default() -> Self {
        // Default values for water at 20°C
        Self {
            viscosity: 1.002e-3,
            bulk_viscosity: 2.81e-3,
            thermal_conductivity: 0.598,
            specific_heat_p: 4182.0,
            specific_heat_v: 4182.0, // Approximately same for water
            density: 998.2,
            sound_speed: 1482.0,
        }
    }
}

/// Stokes absorption model for viscous fluids
#[derive(Debug, Clone))]
pub struct StokesAbsorption {
    params: StokesParameters,
}

impl StokesAbsorption {
    /// Create a new Stokes absorption model
    pub fn new(params: StokesParameters) -> Self {
        Self { params }
    }

    /// Create model for water at given temperature
    pub fn water_at_temperature(temp_celsius: f64) -> Self {
        // Temperature-dependent properties for water
        // Using empirical formulas

        let t = temp_celsius;

        // Dynamic viscosity (Vogel equation)
        let viscosity = 1.002e-3 * (20.0 / (t + 1.0)).exp();

        // Sound speed (Bilaniuk & Wong 1993)
        let sound_speed = 1402.385 + 5.038813 * t - 5.799136e-2 * t * t + 3.287156e-4 * t.powi(3)
            - 1.398845e-6 * t.powi(4)
            + 2.787860e-9 * t.powi(5);

        // Density (IAPWS formulation, simplified)
        let density = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t * t + 1.001685e-4 * t.powi(3)
            - 1.120083e-6 * t.powi(4)
            + 6.536332e-9 * t.powi(5);

        let params = StokesParameters {
            viscosity,
            bulk_viscosity: viscosity * 2.8, // Approximate ratio for water
            thermal_conductivity: 0.598 * (1.0 + 0.003 * (t - 20.0)),
            specific_heat_p: 4182.0,
            specific_heat_v: 4182.0,
            density,
            sound_speed,
        };

        Self::new(params)
    }

    /// Calculate classical (Stokes) absorption coefficient
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let p = &self.params;

        // Classical absorption: α = ω²/(2ρc³) * [4μ/3 + μ_B + κ(γ-1)²/Cp]
        // where μ is shear viscosity, μ_B is bulk viscosity, κ is thermal conductivity

        let viscous_term = (4.0 * p.viscosity / 3.0) + p.bulk_viscosity;

        let gamma = p.specific_heat_p / p.specific_heat_v;
        let thermal_term = p.thermal_conductivity * (gamma - 1.0).powi(2) / p.specific_heat_p;

        omega.powi(2) / (2.0 * p.density * p.sound_speed.powi(3)) * (viscous_term + thermal_term)
    }

    /// Calculate relaxation absorption (for frequencies where relaxation occurs)
    pub fn relaxation_absorption(&self, frequency: f64, relaxation_freq: f64) -> f64 {
        // Relaxation absorption: α_r = A * f² / (f² + f_r²)
        // where f_r is the relaxation frequency

        let classical = self.absorption_at_frequency(frequency);
        let f_ratio = frequency / relaxation_freq;

        classical * f_ratio.powi(2) / (1.0 + f_ratio.powi(2))
    }

    /// Get total absorption including relaxation effects
    pub fn total_absorption(&self, frequency: f64) -> f64 {
        // For water, main relaxation frequencies are:
        // - Structural relaxation: ~1 GHz
        // - Thermal relaxation: ~100 MHz (depends on temperature)

        let classical = self.absorption_at_frequency(frequency);

        // Add relaxation contributions if needed
        // This is simplified; real implementation would need multiple relaxation processes
        if frequency > 1e6 {
            // Add structural relaxation contribution
            let structural = self.relaxation_absorption(frequency, 1e9);
            classical + structural * 0.1 // Weight factor
        } else {
            classical
        }
    }
}
