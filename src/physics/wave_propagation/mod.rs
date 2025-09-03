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

use crate::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
use crate::error::{KwaversError, KwaversResult, PhysicsError};
use ndarray::{Array2, Array3};
use std::f64::consts::PI;

pub mod fresnel;
pub mod interface;
pub mod reflection;
pub mod refraction;
pub mod scattering;
pub mod snell; // Unified scattering module

pub use fresnel::{FresnelCalculator, FresnelCoefficients};
pub use interface::{InterfaceProperties, InterfaceType};
pub use reflection::{ReflectionCalculator, ReflectionCoefficients};
pub use refraction::{RefractionAngles, RefractionCalculator};
pub use scattering::{PhaseFunction, ScatteringCalculator, ScatteringRegime, VolumeScattering};
pub use snell::{CriticalAngles, SnellLawCalculator};

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
    #[must_use]
    pub fn water() -> Self {
        Self {
            wave_speed: SOUND_SPEED_WATER,
            density: DENSITY_WATER,
            refractive_index: 1.333, // At 20°C, 589 nm
            absorption: 0.0,
            anisotropy: None,
        }
    }

    /// Calculate acoustic impedance Z = ρc
    #[must_use]
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.wave_speed
    }

    /// Calculate optical impedance Z = √(μ/ε) ≈ Z₀/n for non-magnetic media
    #[must_use]
    pub fn optical_impedance(&self) -> f64 {
        const VACUUM_IMPEDANCE: f64 = 376.730313668; // Ohms
        VACUUM_IMPEDANCE / self.refractive_index
    }
}

/// Main wave propagation calculator
#[derive(Debug)]
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
    #[must_use]
    pub fn energy_reflection(&self) -> f64 {
        self.reflection_amplitude.powi(2)
    }

    /// Calculate energy transmission coefficient T = |t|²
    #[must_use]
    pub fn energy_transmission(&self) -> f64 {
        if self.total_internal_reflection {
            0.0
        } else {
            self.transmission_amplitude.powi(2)
        }
    }

    /// Verify energy conservation: R + T = 1 (for lossless media)
    #[must_use]
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

        let coeffs = calc
            .calculate_coefficients(PI / 6.0, Some(Polarization::Unpolarized))
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
            let coeffs = calc
                .calculate_coefficients(angle, None)
                .expect("Failed to calculate coefficients");

            // Energy should be conserved (within numerical precision)
            let energy_sum = coeffs.verify_energy_conservation();
            assert!(
                (energy_sum - 1.0).abs() < 1e-10,
                "Energy not conserved at angle {}: sum = {}",
                angle,
                energy_sum
            );
        }
    }
}
