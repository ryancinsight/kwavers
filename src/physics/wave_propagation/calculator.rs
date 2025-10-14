//! Main wave propagation calculator
//!
//! This module provides the primary wave propagation calculator that coordinates
//! calculations across different wave modes and interface types.

use crate::error::{KwaversError, KwaversResult, PhysicsError};
use crate::physics::wave_propagation::{
    coefficients::PropagationCoefficients, fresnel::FresnelCalculator, interface::Interface,
    snell::SnellLawCalculator, Polarization, WaveMode,
};
use std::f64::consts::PI;

/// Main wave propagation calculator
#[derive(Debug)]
pub struct WavePropagationCalculator {
    /// Wave mode
    mode: WaveMode,
    /// Interface configuration
    interface: Interface,
    /// Frequency \[Hz\]
    frequency: f64,
    /// Wavelength \[m\]
    wavelength: f64,
}

impl WavePropagationCalculator {
    /// Create a new calculator
    #[must_use]
    pub fn new(mode: WaveMode, interface: Interface, frequency: f64) -> Self {
        let wavelength = interface.medium1.wave_speed / frequency;
        Self {
            mode,
            interface,
            frequency,
            wavelength,
        }
    }

    /// Get the wave number (k = 2π/λ)
    #[must_use]
    pub fn wave_number(&self) -> f64 {
        2.0 * PI / self.wavelength
    }

    /// Calculate the Fresnel reflection coefficient for normal incidence
    #[must_use]
    pub fn normal_reflection_coefficient(&self) -> f64 {
        let z1 = self.interface.medium1.impedance();
        let z2 = self.interface.medium2.impedance();
        ((z2 - z1) / (z2 + z1)).abs()
    }

    /// Check if frequency is appropriate for the wavelength assumption
    #[must_use]
    pub fn validate_frequency_wavelength_consistency(&self) -> bool {
        let expected_wavelength = self.interface.medium1.wave_speed / self.frequency;
        (self.wavelength - expected_wavelength).abs() < 1e-10
    }

    /// Calculate reflection and transmission for given incident angle
    pub fn calculate_coefficients(
        &self,
        incident_angle: f64,
        polarization: Option<Polarization>,
    ) -> KwaversResult<PropagationCoefficients> {
        // Validate incident angle
        if !(0.0..=PI / 2.0).contains(&incident_angle) {
            return Err(KwaversError::Physics(PhysicsError::InvalidState {
                field: "incident_angle".to_string(),
                value: format!("{incident_angle}"),
                reason: "must be between 0 and π/2".to_string(),
            }));
        }

        // Calculate transmitted angle using Snell's law
        let snell_calc = SnellLawCalculator::new(&self.interface);
        let transmitted_angle = snell_calc.calculate_transmitted_angle(incident_angle)?;

        // Check for total internal reflection
        let critical_angle = snell_calc.critical_angle();
        let total_internal_reflection =
            critical_angle.is_some() && incident_angle > critical_angle.unwrap();

        // Calculate coefficients based on wave mode
        let coefficients = match self.mode {
            WaveMode::Acoustic => {
                self.calculate_acoustic_coefficients(incident_angle, transmitted_angle)?
            }
            WaveMode::Optical => {
                let pol = polarization.unwrap_or(Polarization::Unpolarized);
                self.calculate_optical_coefficients(incident_angle, transmitted_angle, pol)?
            }
            WaveMode::ElasticShear | WaveMode::ElasticCompressional => {
                self.calculate_elastic_coefficients(incident_angle, transmitted_angle)?
            }
        };

        // Get impedances for acoustic mode (for energy conservation validation)
        let (impedance1, impedance2) = if matches!(self.mode, WaveMode::Acoustic) {
            (
                Some(self.interface.medium1.acoustic_impedance()),
                Some(self.interface.medium2.acoustic_impedance()),
            )
        } else {
            (None, None)
        };

        Ok(PropagationCoefficients {
            reflection_amplitude: coefficients.0,
            transmission_amplitude: coefficients.1,
            reflection_phase: coefficients.2,
            transmission_phase: coefficients.3,
            total_internal_reflection,
            incident_angle,
            transmitted_angle: if total_internal_reflection {
                None
            } else {
                Some(transmitted_angle)
            },
            impedance1,
            impedance2,
        })
    }

    /// Calculate acoustic reflection and transmission coefficients
    fn calculate_acoustic_coefficients(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> KwaversResult<(f64, f64, f64, f64)> {
        let z1 = self.interface.medium1.acoustic_impedance();
        let z2 = self.interface.medium2.acoustic_impedance();

        let cos_i = incident_angle.cos();
        let cos_t = transmitted_angle.cos();

        // Acoustic reflection coefficient (pressure)
        let r = (z2 * cos_i - z1 * cos_t) / (z2 * cos_i + z1 * cos_t);

        // Acoustic transmission coefficient (pressure)
        let t = (2.0 * z2 * cos_i) / (z2 * cos_i + z1 * cos_t);

        // Phase shifts
        let r_phase = if r < 0.0 { PI } else { 0.0 };
        let t_phase = 0.0; // No phase shift for transmission

        Ok((r.abs(), t.abs(), r_phase, t_phase))
    }

    /// Calculate optical Fresnel coefficients
    fn calculate_optical_coefficients(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
        polarization: Polarization,
    ) -> KwaversResult<(f64, f64, f64, f64)> {
        let n1 = self.interface.medium1.refractive_index;
        let n2 = self.interface.medium2.refractive_index;

        let fresnel = FresnelCalculator::new(n1, n2);
        let coeffs = fresnel.calculate(incident_angle, transmitted_angle, polarization)?;

        Ok((
            coeffs.reflection_amplitude,
            coeffs.transmission_amplitude,
            coeffs.reflection_phase,
            coeffs.transmission_phase,
        ))
    }

    /// Calculate elastic wave coefficients (simplified implementation)
    fn calculate_elastic_coefficients(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> KwaversResult<(f64, f64, f64, f64)> {
        // For elastic waves, use acoustic impedance as approximation
        // This is a simplified implementation - full elastic analysis would require
        // considering P-wave and S-wave velocities and mode conversion
        self.calculate_acoustic_coefficients(incident_angle, transmitted_angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::wave_propagation::{
        interface::{Interface, InterfaceType},
        medium::MediumProperties,
    };

    fn create_test_interface() -> Interface {
        Interface {
            medium1: MediumProperties::water(),
            medium2: MediumProperties {
                wave_speed: 4000.0,
                density: 2000.0,
                refractive_index: 1.5,
                absorption: 0.0,
                anisotropy: None,
            },
            normal: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            interface_type: InterfaceType::Planar,
        }
    }

    #[test]
    fn test_calculator_creation() {
        let interface = create_test_interface();
        let calc = WavePropagationCalculator::new(WaveMode::Acoustic, interface, 1000.0);

        assert!(calc.validate_frequency_wavelength_consistency());
        assert!(calc.wave_number() > 0.0);
    }

    #[test]
    fn test_normal_incidence() {
        let interface = create_test_interface();
        let calc = WavePropagationCalculator::new(WaveMode::Acoustic, interface, 1000.0);

        let coeffs = calc.calculate_coefficients(0.0, None).unwrap();

        // Energy conservation check
        let error = coeffs.energy_conservation_error();
        assert!(error < 1e-10, "Energy conservation error: {}", error);

        // No total internal reflection at normal incidence
        assert!(!coeffs.total_internal_reflection);
        assert!(coeffs.transmitted_angle.is_some());
    }
}
