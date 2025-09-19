//! Wave propagation attenuation calculations
//!
//! This module implements attenuation calculations for wave propagation
//! in absorbing media, including Beer-Lambert law and frequency-dependent absorption.

use ndarray::Array3;
use std::f64::consts::PI;

/// Attenuation calculator for wave propagation in absorbing media
#[derive(Debug)]
pub struct AttenuationCalculator {
    /// Absorption coefficient [1/m] or [Np/m]
    absorption_coefficient: f64,
    /// Frequency [Hz]
    frequency: f64,
    /// Wave speed [m/s]
    wave_speed: f64,
}

impl AttenuationCalculator {
    /// Create a new attenuation calculator
    #[must_use]
    pub fn new(absorption_coefficient: f64, frequency: f64, wave_speed: f64) -> Self {
        Self {
            absorption_coefficient,
            frequency,
            wave_speed,
        }
    }

    /// Calculate amplitude attenuation over distance using Beer-Lambert law
    /// A(x) = A₀ * exp(-α * x)
    #[must_use]
    pub fn amplitude_at_distance(&self, initial_amplitude: f64, distance: f64) -> f64 {
        initial_amplitude * (-self.absorption_coefficient * distance).exp()
    }

    /// Calculate intensity attenuation (intensity ~ amplitude²)
    /// I(x) = I₀ * exp(-2α * x)
    #[must_use]
    pub fn intensity_at_distance(&self, initial_intensity: f64, distance: f64) -> f64 {
        initial_intensity * (-2.0 * self.absorption_coefficient * distance).exp()
    }

    /// Calculate attenuation in dB over distance
    /// `Attenuation_dB` = 20 * log₁₀(A₀/A) = 8.686 * α * x
    #[must_use]
    pub fn attenuation_db(&self, distance: f64) -> f64 {
        8.686 * self.absorption_coefficient * distance
    }

    /// Calculate frequency-dependent absorption for this calculator's frequency
    #[must_use]
    pub fn frequency_dependent_absorption(&self, alpha_0: f64, power_law: f64) -> f64 {
        alpha_0 * self.frequency.powf(power_law)
    }
    
    /// Calculate wave number (k = 2π/λ = 2πf/c)
    #[must_use]  
    pub fn wave_number(&self) -> f64 {
        2.0 * PI * self.frequency / self.wave_speed
    }
    
    /// Calculate penetration depth (distance where amplitude drops to 1/e)
    #[must_use]
    pub fn penetration_depth(&self) -> f64 {
        1.0 / self.absorption_coefficient
    }

    /// Calculate frequency-dependent absorption for acoustic waves in tissue
    /// α = α₀ * f^n where n is typically 1-2
    #[must_use]
    pub fn tissue_absorption(frequency: f64, alpha_0: f64, power_law: f64) -> f64 {
        alpha_0 * frequency.powf(power_law)
    }

    /// Calculate thermo-viscous absorption in fluids (classical absorption)
    /// α = 2πf²/ρc³ * (4μ/3 + `μ_B` + κ(γ-1)/C_p)
    #[must_use]
    pub fn classical_absorption(
        frequency: f64,
        density: f64,
        sound_speed: f64,
        shear_viscosity: f64,
        bulk_viscosity: f64,
        thermal_conductivity: f64,
        specific_heat_ratio: f64,
        specific_heat_cp: f64,
    ) -> f64 {
        let omega = 2.0 * PI * frequency;
        let factor1 = omega * omega / (2.0 * density * sound_speed.powi(3));
        let viscous_term = (4.0 / 3.0) * shear_viscosity + bulk_viscosity;
        let thermal_term = thermal_conductivity * (specific_heat_ratio - 1.0) / specific_heat_cp;
        factor1 * (viscous_term + thermal_term)
    }

    /// Apply attenuation to a 3D field
    pub fn apply_attenuation_field(
        &self,
        field: &mut Array3<f64>,
        source_position: [f64; 3],
        grid_spacing: [f64; 3],
    ) {
        let (nx, ny, nz) = field.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * grid_spacing[0];
                    let y = j as f64 * grid_spacing[1];
                    let z = k as f64 * grid_spacing[2];

                    let distance = ((x - source_position[0]).powi(2)
                        + (y - source_position[1]).powi(2)
                        + (z - source_position[2]).powi(2))
                    .sqrt();

                    field[(i, j, k)] *= (-self.absorption_coefficient * distance).exp();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_amplitude_attenuation() {
        let calc = AttenuationCalculator::new(0.1, 1000.0, 1500.0);
        let initial = 1.0;
        let distance = 10.0;
        let attenuated = calc.amplitude_at_distance(initial, distance);
        let expected = initial * (-0.1 * 10.0).exp();
        assert_relative_eq!(attenuated, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_penetration_depth() {
        let calc = AttenuationCalculator::new(0.1, 1000.0, 1500.0);
        let depth = calc.penetration_depth();
        assert_relative_eq!(depth, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_attenuation_db() {
        let calc = AttenuationCalculator::new(0.1, 1000.0, 1500.0);
        let distance = 10.0;
        let db = calc.attenuation_db(distance);
        let expected = 8.686 * 0.1 * 10.0;
        assert_relative_eq!(db, expected, epsilon = 1e-10);
    }
}