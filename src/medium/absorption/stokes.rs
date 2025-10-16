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
#[derive(Debug, Clone)]
pub struct StokesAbsorption {
    params: StokesParameters,
}

impl StokesAbsorption {
    /// Create a new Stokes absorption model
    #[must_use]
    pub fn new(params: StokesParameters) -> Self {
        Self { params }
    }

    /// Create model for water at given temperature
    #[must_use]
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

        // Density (IAPWS-95 formulation for water properties)
        // Full formulation uses critical parameters and dimensionless Helmholtz energy
        // Simplified formulation valid for 0-100°C at atmospheric pressure
        //
        // References:
        // - Wagner & Pruß (2002): "IAPWS formulation 1995"
        // - Lemmon et al. (2005): "Thermodynamic properties of water"
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
    #[must_use]
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
    #[must_use]
    pub fn relaxation_absorption(&self, frequency: f64, relaxation_freq: f64) -> f64 {
        // Relaxation absorption: α_r = A * f² / (f² + f_r²)
        // where f_r is the relaxation frequency

        let classical = self.absorption_at_frequency(frequency);
        let f_ratio = frequency / relaxation_freq;

        classical * f_ratio.powi(2) / (1.0 + f_ratio.powi(2))
    }

    /// Get total absorption including multiple relaxation effects
    ///
    /// Implements multi-relaxation absorption model for water:
    /// α = α_classical + ∑ᵢ Aᵢ (f/fᵢ)² / (1 + (f/fᵢ)²)
    ///
    /// Main relaxation processes in water:
    /// - Structural relaxation: ~1 GHz (molecular rearrangement)
    /// - Thermal relaxation: ~100 MHz (thermal diffusion)
    /// - Viscous relaxation: ~10 MHz (shear viscosity)
    ///
    /// References:
    /// - Pinkerton (1949): "The absorption of ultrasonic waves in liquids"
    /// - Litovitz & Davis (1965): "Structural and shear relaxation in liquids"
    /// - Slie et al. (1966): "Ultrasonic shear and compressional relaxation in liquids"
    #[must_use]
    pub fn total_absorption(&self, frequency: f64) -> f64 {
        let classical = self.absorption_at_frequency(frequency);

        // Relaxation parameters for water at room temperature
        struct RelaxationMode {
            freq: f64,      // Relaxation frequency [Hz]
            amplitude: f64, // Amplitude factor relative to classical
        }
        
        let modes = [
            RelaxationMode { freq: 1e9,  amplitude: 0.15 }, // Structural (dominant)
            RelaxationMode { freq: 1e8,  amplitude: 0.08 }, // Thermal
            RelaxationMode { freq: 1e7,  amplitude: 0.03 }, // Viscous
        ];
        
        // Sum all relaxation contributions
        let mut relaxation_total = 0.0;
        for mode in &modes {
            let f_ratio = frequency / mode.freq;
            let relaxation_contrib = classical * mode.amplitude * f_ratio.powi(2) / (1.0 + f_ratio.powi(2));
            relaxation_total += relaxation_contrib;
        }

        classical + relaxation_total
    }
}
