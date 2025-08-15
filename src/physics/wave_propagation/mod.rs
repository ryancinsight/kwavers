//! Wave propagation physics: reflection, refraction, and transmission
//!
//! This module implements fundamental wave propagation phenomena at interfaces
//! between media with different properties, following Snell's law and Fresnel equations.
//!
//! # Literature References
//!
//! 1. **Born, M., & Wolf, E. (1999)**. "Principles of Optics" (7th ed.). 
//!    Cambridge University Press. ISBN: 978-0521642224
//!    - Comprehensive treatment of reflection and refraction
//!    - Fresnel equations derivation and applications
//!
//! 2. **Kinsler, L. E., et al. (2000)**. "Fundamentals of Acoustics" (4th ed.).
//!    Wiley. ISBN: 978-0471847892
//!    - Acoustic reflection and transmission coefficients
//!    - Mode conversion at interfaces
//!
//! 3. **Brekhovskikh, L. M., & Lysanov, Y. P. (2003)**. "Fundamentals of Ocean Acoustics"
//!    (3rd ed.). Springer. ISBN: 978-0387954677
//!    - Reflection from layered media
//!    - Critical angles and total internal reflection
//!
//! 4. **Pierce, A. D. (2019)**. "Acoustics: An Introduction to Its Physical 
//!    Principles and Applications" (3rd ed.). Springer. ISBN: 978-3030112134
//!    - Comprehensive acoustic wave propagation theory

use crate::error::{KwaversResult, KwaversError, PhysicsError};
use crate::constants::physics::{SOUND_SPEED_WATER, DENSITY_WATER};
use ndarray::{Array1, Array2, Array3, Axis, Zip};
use std::f64::consts::PI;

pub mod reflection;
pub mod refraction;
pub mod interface;
pub mod fresnel;
pub mod snell;
pub mod scattering; // Unified scattering module

pub use reflection::{ReflectionCalculator, ReflectionCoefficients};
pub use refraction::{RefractionCalculator, RefractionAngles};
pub use interface::{InterfaceProperties, InterfaceType};
pub use fresnel::{FresnelCoefficients, FresnelCalculator};
pub use snell::{SnellLawCalculator, CriticalAngles};
pub use scattering::{ScatteringCalculator, ScatteringRegime, VolumeScattering, PhaseFunction};

/// Wave propagation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveMode {
    /// Acoustic pressure wave
    Acoustic,
    /// Optical electromagnetic wave  
    Optical,
    /// Elastic shear wave
    ElasticShear,
    /// Elastic compressional wave
    ElasticCompressional,
}

/// Polarization state for electromagnetic waves
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Transverse electric (S-polarized)
    TransverseElectric,
    /// Transverse magnetic (P-polarized)
    TransverseMagnetic,
    /// Unpolarized
    Unpolarized,
    /// Circular polarization
    Circular,
    /// Elliptical polarization
    Elliptical,
}

/// Interface configuration between two media
#[derive(Debug, Clone)]
pub struct Interface {
    /// Properties of medium 1 (incident side)
    pub medium1: MediumProperties,
    /// Properties of medium 2 (transmission side)
    pub medium2: MediumProperties,
    /// Interface normal vector (unit vector)
    pub normal: [f64; 3],
    /// Interface position
    pub position: [f64; 3],
    /// Interface type
    pub interface_type: InterfaceType,
}

/// Medium properties for wave propagation
#[derive(Debug, Clone)]
pub struct MediumProperties {
    /// Wave speed in the medium [m/s]
    pub wave_speed: f64,
    /// Density for acoustic waves [kg/m³]
    pub density: f64,
    /// Refractive index for optical waves
    pub refractive_index: f64,
    /// Absorption coefficient [1/m]
    pub absorption: f64,
    /// Anisotropy tensor (for anisotropic media)
    pub anisotropy: Option<Array2<f64>>,
}

impl MediumProperties {
    /// Create properties for water at standard conditions
    pub fn water() -> Self {
        Self {
            wave_speed: SOUND_SPEED_WATER,
            density: DENSITY_WATER,
            refractive_index: 1.333,  // At 20°C, 589 nm
            absorption: 0.0,
            anisotropy: None,
        }
    }
    
    /// Calculate acoustic impedance Z = ρc
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.wave_speed
    }
    
    /// Calculate optical impedance Z = √(μ/ε) ≈ Z₀/n for non-magnetic media
    pub fn optical_impedance(&self) -> f64 {
        const VACUUM_IMPEDANCE: f64 = 376.730313668; // Ohms
        VACUUM_IMPEDANCE / self.refractive_index
    }
}

/// Main wave propagation calculator
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
        let wavelength = interface.medium1.wave_speed / frequency;
        Self {
            mode,
            interface,
            frequency,
            wavelength,
        }
    }
    
    /// Calculate reflection and transmission for given incident angle
    pub fn calculate_coefficients(
        &self,
        incident_angle: f64,
        polarization: Option<Polarization>,
    ) -> KwaversResult<PropagationCoefficients> {
        // Validate incident angle
        if incident_angle < 0.0 || incident_angle > PI / 2.0 {
            return Err(KwaversError::Physics(PhysicsError::InvalidState {
                field: "incident_angle".to_string(),
                value: format!("{}", incident_angle),
                reason: "must be between 0 and π/2".to_string(),
            }));
        }
        
        // Calculate transmitted angle using Snell's law
        let snell_calc = SnellLawCalculator::new(&self.interface);
        let transmitted_angle = snell_calc.calculate_transmitted_angle(incident_angle)?;
        
        // Check for total internal reflection
        let critical_angle = snell_calc.critical_angle();
        let total_internal_reflection = critical_angle.is_some() 
            && incident_angle > critical_angle.unwrap();
        
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
        
        Ok(PropagationCoefficients {
            reflection_amplitude: coefficients.0,
            transmission_amplitude: coefficients.1,
            reflection_phase: coefficients.2,
            transmission_phase: coefficients.3,
            total_internal_reflection,
            incident_angle,
            transmitted_angle: if total_internal_reflection { None } else { Some(transmitted_angle) },
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
    
    /// Calculate elastic wave coefficients (simplified for now)
    fn calculate_elastic_coefficients(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> KwaversResult<(f64, f64, f64, f64)> {
        // Use acoustic approximation for elastic waves
        // Full implementation would include mode conversion
        self.calculate_acoustic_coefficients(incident_angle, transmitted_angle)
    }
}

/// Propagation coefficients at an interface
#[derive(Debug, Clone)]
pub struct PropagationCoefficients {
    /// Reflection amplitude coefficient
    pub reflection_amplitude: f64,
    /// Transmission amplitude coefficient
    pub transmission_amplitude: f64,
    /// Reflection phase shift [radians]
    pub reflection_phase: f64,
    /// Transmission phase shift [radians]
    pub transmission_phase: f64,
    /// Whether total internal reflection occurs
    pub total_internal_reflection: bool,
    /// Incident angle [radians]
    pub incident_angle: f64,
    /// Transmitted angle [radians] (None for total internal reflection)
    pub transmitted_angle: Option<f64>,
}

impl PropagationCoefficients {
    /// Calculate energy reflection coefficient R = |r|²
    pub fn energy_reflection(&self) -> f64 {
        self.reflection_amplitude.powi(2)
    }
    
    /// Calculate energy transmission coefficient T = |t|²
    pub fn energy_transmission(&self) -> f64 {
        if self.total_internal_reflection {
            0.0
        } else {
            self.transmission_amplitude.powi(2)
        }
    }
    
    /// Verify energy conservation: R + T = 1 (for lossless media)
    pub fn verify_energy_conservation(&self) -> f64 {
        self.energy_reflection() + self.energy_transmission()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_snells_law() {
        let medium1 = MediumProperties::water();
        let mut medium2 = MediumProperties::water();
        medium2.refractive_index = 1.5; // Glass
        
        let interface = Interface {
            medium1,
            medium2,
            normal: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            interface_type: InterfaceType::Planar,
        };
        
        let calc = WavePropagationCalculator::new(
            WaveMode::Optical,
            interface,
            5e14, // Green light
        );
        
        let coeffs = calc.calculate_coefficients(PI / 6.0, Some(Polarization::Unpolarized))
            .expect("Failed to calculate coefficients");
        
        // Verify Snell's law: n1 * sin(θ1) = n2 * sin(θ2)
        let n1_sin_theta1 = 1.333 * (PI / 6.0).sin();
        let n2_sin_theta2 = 1.5 * coeffs.transmitted_angle.unwrap().sin();
        assert!((n1_sin_theta1 - n2_sin_theta2).abs() < 1e-10);
    }
    
    #[test]
    fn test_energy_conservation() {
        let medium1 = MediumProperties::water();
        let medium2 = MediumProperties::water();
        
        let interface = Interface {
            medium1,
            medium2,
            normal: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            interface_type: InterfaceType::Planar,
        };
        
        let calc = WavePropagationCalculator::new(
            WaveMode::Acoustic,
            interface,
            1e6, // 1 MHz
        );
        
        for angle in [0.0, PI / 6.0, PI / 4.0, PI / 3.0] {
            let coeffs = calc.calculate_coefficients(angle, None)
                .expect("Failed to calculate coefficients");
            
            // Energy should be conserved (within numerical precision)
            let energy_sum = coeffs.verify_energy_conservation();
            assert!((energy_sum - 1.0).abs() < 1e-10,
                "Energy not conserved at angle {}: sum = {}", angle, energy_sum);
        }
    }
}