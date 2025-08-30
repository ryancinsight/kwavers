//! Wave attenuation in absorbing media
//!
//! Implements Beer-Lambert law and frequency-dependent absorption models.

use std::f64::consts::PI;

/// Attenuation calculator for wave propagation in absorbing media
#[derive(Debug, Clone))]
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
    pub fn new(absorption_coefficient: f64, frequency: f64, wave_speed: f64) -> Self {
        Self {
            absorption_coefficient,
            frequency,
            wave_speed,
        }
    }

    /// Calculate amplitude attenuation over distance using Beer-Lambert law
    /// A(x) = A₀ * exp(-α * x)
    pub fn amplitude_at_distance(&self, initial_amplitude: f64, distance: f64) -> f64 {
        initial_amplitude * (-self.absorption_coefficient * distance).exp()
    }

    /// Calculate intensity attenuation (intensity ~ amplitude²)
    /// I(x) = I₀ * exp(-2α * x)
    pub fn intensity_at_distance(&self, initial_intensity: f64, distance: f64) -> f64 {
        initial_intensity * (-2.0 * self.absorption_coefficient * distance).exp()
    }

    /// Calculate attenuation in dB over distance
    /// Attenuation_dB = 20 * log₁₀(A₀/A) = 8.686 * α * x
    pub fn attenuation_db(&self, distance: f64) -> f64 {
        8.686 * self.absorption_coefficient * distance
    }

    /// Calculate frequency-dependent absorption for acoustic waves in tissue
    /// α = α₀ * f^n where n is typically 1-2
    pub fn tissue_absorption(frequency: f64, alpha_0: f64, power_law: f64) -> f64 {
        alpha_0 * frequency.powf(power_law)
    }

    /// Calculate thermo-viscous absorption in fluids (classical absorption)
    /// α = 2πf²/ρc³ * (4μ/3 + μ_B + κ(γ-1)/C_p)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beer_lambert_law() {
        let calc = AttenuationCalculator::new(0.1, 1e6, 1500.0);
        let amplitude = calc.amplitude_at_distance(1.0, 10.0);
        assert_relative_eq!(amplitude, (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_intensity_attenuation() {
        let calc = AttenuationCalculator::new(0.1, 1e6, 1500.0);
        let intensity = calc.intensity_at_distance(1.0, 10.0);
        assert_relative_eq!(intensity, (-2.0_f64).exp(), epsilon = 1e-10);
    }
}
