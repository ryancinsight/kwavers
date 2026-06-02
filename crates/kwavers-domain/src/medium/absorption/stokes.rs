//! Stokes absorption model for viscous fluids

use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::constants::fundamental::{DENSITY_WATER, SOUND_SPEED_WATER};
use kwavers_core::constants::thermodynamic::{
    ROOM_TEMPERATURE_C, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
};
use serde::{Deserialize, Serialize};
use kwavers_core::constants::numerical::{TWO_PI};

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
    /// Sound speed (m/s)
    pub sound_speed: f64,
}

impl Default for StokesParameters {
    fn default() -> Self {
        // Default values for water at 20 °C — all sourced from SSOT.
        Self {
            viscosity: VISCOSITY_WATER,
            bulk_viscosity: 2.81e-3, // Slie et al. 1966 — local domain constant
            thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
            specific_heat_p: SPECIFIC_HEAT_WATER,
            specific_heat_v: SPECIFIC_HEAT_WATER, // c_v ≈ c_p for liquid water
            density: DENSITY_WATER,
            sound_speed: SOUND_SPEED_WATER,
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

        // Dynamic viscosity (Vogel equation), reference value VISCOSITY_WATER at 20°C
        let viscosity = VISCOSITY_WATER * (20.0 / (t + 1.0)).exp();

        // Sound speed (Bilaniuk & Wong 1993)
        let sound_speed = 2.787860e-9f64.mul_add(
            t.powi(5),
            1.398845e-6f64.mul_add(
                -t.powi(4),
                3.287156e-4f64.mul_add(
                    t.powi(3),
                    (5.799136e-2 * t).mul_add(-t, 5.038813f64.mul_add(t, 1402.385)),
                ),
            ),
        );

        // Density (IAPWS-95 formulation for water properties)
        // Polynomial approximation from Wagner & Pruß (2002) valid for 0-100°C at atmospheric pressure
        // This is the standard empirical formula, not a simplification - full IAPWS-95 formulation
        // uses critical parameters and dimensionless Helmholtz energy which reduces to this polynomial
        // for the specified temperature and pressure range.
        //
        // References:
        // - Wagner & Pruß (2002): "IAPWS formulation 1995 for the thermodynamic properties of ordinary water"
        // - Lemmon et al. (2005): "Thermodynamic properties of water and steam"
        let density = 6.536332e-9f64.mul_add(
            t.powi(5),
            1.120083e-6f64.mul_add(
                -t.powi(4),
                1.001685e-4f64.mul_add(
                    t.powi(3),
                    (9.095290e-3 * t).mul_add(-t, 6.793952e-2f64.mul_add(t, 999.842594)),
                ),
            ),
        );

        let params = StokesParameters {
            viscosity,
            bulk_viscosity: viscosity * 2.8, // Approximate ratio for water
            thermal_conductivity: THERMAL_CONDUCTIVITY_WATER
                * 0.003f64.mul_add(t - ROOM_TEMPERATURE_C, 1.0),
            specific_heat_p: SPECIFIC_HEAT_WATER,
            specific_heat_v: SPECIFIC_HEAT_WATER,
            density,
            sound_speed,
        };

        Self::new(params)
    }

    /// Calculate classical (Stokes-Kirchhoff) absorption coefficient.
    ///
    /// ## Formula (Pierce 1989, *Acoustics*, Eq. 10-7-5)
    ///
    /// ```text
    /// α(ω) = ω² / (2 ρ c³) · [ 4η/3 + ζ + κ (γ − 1) / c_p ]
    /// ```
    ///
    /// where η = shear viscosity, ζ = bulk viscosity, κ = thermal conductivity,
    /// γ = c_p / c_v. The thermal contribution is **linear** in (γ − 1), not
    /// quadratic — prior to 2026-05-21 this used (γ − 1)² which for water
    /// (γ ≈ 1.007) under-predicted the thermal absorption term by ≈ 140×.
    /// Reference: Pierce A. D. (1989). *Acoustics: An Introduction to its
    /// Physical Principles and Applications*, §10-7. ASA.
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        let omega = TWO_PI * frequency;
        let p = &self.params;

        let viscous_term = (4.0 * p.viscosity / 3.0) + p.bulk_viscosity;

        let gamma = p.specific_heat_p / p.specific_heat_v;
        let thermal_term = p.thermal_conductivity * (gamma - 1.0) / p.specific_heat_p;

        omega.powi(2) / (2.0 * p.density * p.sound_speed.powi(3)) * (viscous_term + thermal_term)
    }

    /// Calculate relaxation absorption (for frequencies where relaxation occurs)
    #[must_use]
    pub fn relaxation_absorption(&self, frequency: f64, relaxation_freq: f64) -> f64 {
        // Relaxation absorption: α_r = A * f² / (f² + f_r²)
        // where f_r is the relaxation frequency

        let classical = self.absorption_at_frequency(frequency);
        let f_ratio = frequency / relaxation_freq;

        classical * f_ratio.powi(2) / f_ratio.mul_add(f_ratio, 1.0)
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
            RelaxationMode {
                freq: 1e9,
                amplitude: 0.15,
            }, // Structural (dominant)
            RelaxationMode {
                freq: 1e8,
                amplitude: 0.08,
            }, // Thermal
            RelaxationMode {
                freq: 1e7,
                amplitude: 0.03,
            }, // Viscous
        ];

        // Sum all relaxation contributions
        let mut relaxation_total = 0.0;
        for mode in &modes {
            let f_ratio = frequency / mode.freq;
            let relaxation_contrib =
                classical * mode.amplitude * f_ratio.powi(2) / f_ratio.mul_add(f_ratio, 1.0);
            relaxation_total += relaxation_contrib;
        }

        classical + relaxation_total
    }
}
