//! Main wave propagation calculator
//!
//! Coordinates reflection, refraction, and transmission calculations.

use super::fresnel::FresnelCoefficients;
use super::reflection::ReflectionCoefficients;
use super::snell::SnellLawCalculator;
use super::{Interface, WaveMode};
use crate::error::KwaversResult;

/// Main wave propagation calculator
#[derive(Debug, Clone)]
pub struct WavePropagationCalculator {
    /// Wave mode
    mode: WaveMode,
    /// Interface configuration
    interface: Interface,
    /// Frequency [Hz]
    frequency: f64,
    /// Wavelength [m]
    wavelength: f64,
}

impl WavePropagationCalculator {
    /// Create a new calculator
    pub fn new(mode: WaveMode, interface: Interface, frequency: f64) -> Self {
        let wavelength = match mode {
            WaveMode::Acoustic => interface.medium1.wave_speed / frequency,
            WaveMode::Optical => 299792458.0 / (frequency * interface.medium1.refractive_index),
        };

        Self {
            mode,
            interface,
            frequency,
            wavelength,
        }
    }

    /// Calculate reflection and transmission at interface
    pub fn calculate_interface_response(
        &self,
        incident_angle: f64,
    ) -> KwaversResult<InterfaceResponse> {
        let snell = SnellLawCalculator::new(&self.interface);

        let transmitted_angle = snell.calculate_transmitted_angle(incident_angle)?;

        let coefficients = match self.mode {
            WaveMode::Acoustic => {
                let refl = ReflectionCoefficients::acoustic(
                    self.interface.medium1.acoustic_impedance(),
                    self.interface.medium2.acoustic_impedance(),
                    incident_angle,
                    transmitted_angle,
                );
                InterfaceCoefficients::Acoustic(refl)
            }
            WaveMode::Optical => {
                let fresnel = FresnelCoefficients::calculate(
                    self.interface.medium1.refractive_index,
                    self.interface.medium2.refractive_index,
                    incident_angle,
                )?;
                InterfaceCoefficients::Optical(fresnel)
            }
        };

        Ok(InterfaceResponse {
            incident_angle,
            transmitted_angle,
            coefficients,
            total_internal_reflection: transmitted_angle.is_nan(),
        })
    }
}

/// Response at an interface
#[derive(Debug, Clone)]
pub struct InterfaceResponse {
    /// Incident angle [rad]
    pub incident_angle: f64,
    /// Transmitted angle [rad]
    pub transmitted_angle: f64,
    /// Reflection/transmission coefficients
    pub coefficients: InterfaceCoefficients,
    /// Whether total internal reflection occurs
    pub total_internal_reflection: bool,
}

/// Interface coefficients by mode
#[derive(Debug, Clone)]
pub enum InterfaceCoefficients {
    Acoustic(ReflectionCoefficients),
    Optical(FresnelCoefficients),
}
