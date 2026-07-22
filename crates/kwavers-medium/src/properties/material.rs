//! Unified Material Properties (Consolidated from physics/materials)
//!
//! This module provides a unified `AcousticMaterialProperties` struct that combines all acoustic, thermal,
//! optical, and perfusion properties needed for multi-physics simulations.
//!
//! This replaces the `physics/materials/mod.rs` implementation, moving material property
//! definitions from the physics layer to the domain layer where they belong (layer fix).
//!
//! # Architecture
//!
//! Material properties are **domain specifications** (WHAT materials have), not physics equations
//! (HOW materials behave). Therefore, they belong in the domain layer.
//!
//! - **Domain Layer** (`domain/medium/properties/`): Material specifications ✅ CORRECT
//! - **Physics Layer**: Uses these properties in equations, doesn't define them ✅ CORRECT
//!
//! # Physical Foundation
//!
//! Properties are sourced from:
//! - Duck (1990) - Physical Properties of Tissues
//! - Perry & Green (2007) - Chemical Engineering Handbook
//! - IEC 61161:2013 - Ultrasound equipment safety
//! - FDA ultrasound safety guidelines
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers_medium::properties::{AcousticMaterialProperties, tissue, fluids, implants};
//!
//! // Tissue properties
//! let brain = tissue::BRAIN_WHITE_MATTER;
//! assert!(brain.validate().is_ok());
//!
//! // Fluid properties
//! let blood = fluids::WHOLE_BLOOD;
//! let r = brain.reflection_coefficient(&blood);
//!
//! // Implant properties
//! let titanium = implants::TITANIUM_GRADE5;
//! assert!(titanium.sound_speed > 5000.0);
//! ```

use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::thermodynamic::{BODY_TEMPERATURE_C, ROOM_TEMPERATURE_C};
use kwavers_core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Unified material property structure
///
/// Contains all acoustic, thermal, optical, and perfusion properties needed for simulations.
/// This struct serves as the SSOT (Single Source of Truth) for material properties throughout
/// the kwavers library.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AcousticMaterialProperties {
    // ========================================================================
    // Acoustic Properties
    // ========================================================================
    /// Speed of sound (m/s)
    pub sound_speed: f64,

    /// Density [kg/m³]
    pub density: f64,

    /// Acoustic impedance Z = ρc [kg/(m²·s)]
    pub impedance: f64,

    /// Absorption coefficient α₀ [dB/(cm·MHz^y)] (frequency-dependent: α = α₀·f^y)
    ///
    /// Evaluated at frequency f (MHz) gives α(f) = α₀·f^y in dB/cm.
    /// Canonical tissue values from Duck (1990) Table 4.1.
    pub absorption_coefficient: f64,

    /// Absorption frequency exponent (typically 1.0-2.0)
    pub absorption_exponent: f64,

    /// Nonlinearity parameter B/A (Goldberg-Saffren)
    pub nonlinearity_parameter: f64,

    /// Shear viscosity [Pa·s]
    pub shear_viscosity: f64,

    /// Bulk viscosity [Pa·s]
    pub bulk_viscosity: f64,

    // ========================================================================
    // Thermal Properties
    // ========================================================================
    /// Specific heat capacity [J/(kg·K)]
    pub specific_heat: f64,

    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,

    /// Thermal diffusivity α = k/(ρ·c) [m²/s]
    pub thermal_diffusivity: f64,

    // ========================================================================
    // Perfusion Properties (Tissue-Specific)
    // ========================================================================
    /// Blood perfusion rate [mL/100g/min]
    pub perfusion_rate: f64,

    /// Arterial blood temperature [°C]
    pub arterial_temperature: f64,

    /// Metabolic heat generation [W/kg]
    pub metabolic_heat: f64,

    // ========================================================================
    // Optical Properties
    // ========================================================================
    /// Absorption coefficient [1/m]
    pub optical_absorption: f64,

    /// Scattering coefficient [1/m]
    pub optical_scattering: f64,

    /// Refractive index [-]
    pub refractive_index: f64,

    // ========================================================================
    // State Reference
    // ========================================================================
    /// Temperature at which properties are defined [°C]
    pub reference_temperature: f64,

    /// Pressure at which properties are defined (Pa)
    pub reference_pressure: f64,
}

/// Pressure-amplitude reflection coefficient at a normal-incidence planar
/// interface (workspace SSOT): `R = (Z_t − Z_i) / (Z_t + Z_i)` for a wave
/// travelling from a medium of acoustic impedance `z_incident` into one of
/// impedance `z_transmitted` [both rayl]. Sign follows the standard convention
/// (positive when transmitting into a higher-impedance medium); take `.abs()`
/// for the magnitude.
#[inline]
#[must_use]
pub fn reflection_coefficient(z_incident: f64, z_transmitted: f64) -> f64 {
    (z_transmitted - z_incident) / (z_transmitted + z_incident)
}

/// Pressure-amplitude transmission coefficient at a normal-incidence planar
/// interface (workspace SSOT): `T = 2 Z_t / (Z_i + Z_t)` [both rayl]. This is
/// the pressure-amplitude companion of [`reflection_coefficient`] and satisfies
/// the exact relation `T = 1 + R` (so `T` can exceed 1 when transmitting into a
/// higher-impedance medium); the lossless energy balance is
/// `R² + (Z_i/Z_t)·T² = 1`.
#[inline]
#[must_use]
pub fn transmission_coefficient(z_incident: f64, z_transmitted: f64) -> f64 {
    2.0 * z_transmitted / (z_incident + z_transmitted)
}

/// Intensity reflection coefficient at a normal-incidence planar interface
/// (Chapter 1 Corollary 1.2): `R_I = R² = ((Z_t − Z_i)/(Z_t + Z_i))²` — the
/// fraction of incident acoustic intensity reflected. Lossless energy balance:
/// `R_I + T_I = 1`.
#[inline]
#[must_use]
pub fn intensity_reflection_coefficient(z_incident: f64, z_transmitted: f64) -> f64 {
    let r = reflection_coefficient(z_incident, z_transmitted);
    r * r
}

/// Intensity transmission coefficient at a normal-incidence planar interface
/// (Chapter 1 Corollary 1.2): `T_I = 1 − R_I = 4 Z_i Z_t / (Z_i + Z_t)²` — the
/// fraction of incident acoustic intensity transmitted. Returns 0 for a
/// non-positive total impedance.
#[inline]
#[must_use]
pub fn intensity_transmission_coefficient(z_incident: f64, z_transmitted: f64) -> f64 {
    let sum = z_incident + z_transmitted;
    if sum > 0.0 {
        4.0 * z_incident * z_transmitted / (sum * sum)
    } else {
        0.0
    }
}

/// Time-averaged acoustic intensity of a harmonic plane wave (Chapter 1
/// Theorem 1.6): `I = A² / (2 Z) = p_rms² / Z`, for pressure amplitude `A` \`Pa`
/// and specific acoustic impedance `Z = ρ₀c₀` \[rayl]. Returns 0 for non-positive
/// impedance.
#[inline]
#[must_use]
pub fn plane_wave_intensity(pressure_amplitude: f64, impedance: f64) -> f64 {
    if impedance > 0.0 {
        pressure_amplitude * pressure_amplitude / (2.0 * impedance)
    } else {
        0.0
    }
}

/// Coefficient of nonlinearity `β = 1 + B/(2A)` from the parameter of
/// nonlinearity `B/A` (Chapter 3 Definition 3.2) — the workspace SSOT for this
/// conversion. `β` is the coefficient in the Westervelt/Burgers nonlinear term;
/// e.g. water `B/A ≈ 5` ⇒ `β ≈ 3.5`.
#[inline]
#[must_use]
pub fn coefficient_of_nonlinearity(b_over_a: f64) -> f64 {
    1.0 + b_over_a / 2.0
}

impl AcousticMaterialProperties {
    /// Create material properties with core parameters
    #[must_use]
    pub fn new(
        sound_speed: f64,
        density: f64,
        absorption_coefficient: f64,
        specific_heat: f64,
        thermal_conductivity: f64,
    ) -> Self {
        let impedance = density * sound_speed;
        let thermal_diffusivity = thermal_conductivity / (density * specific_heat);

        Self {
            sound_speed,
            density,
            impedance,
            absorption_coefficient,
            absorption_exponent: 1.0,
            nonlinearity_parameter: 0.0,
            shear_viscosity: VISCOSITY_WATER,
            bulk_viscosity: VISCOSITY_WATER,
            specific_heat,
            thermal_conductivity,
            thermal_diffusivity,
            perfusion_rate: 0.0,
            arterial_temperature: BODY_TEMPERATURE_C,
            metabolic_heat: 0.0,
            optical_absorption: 0.0,
            optical_scattering: 0.0,
            refractive_index: 1.0,
            reference_temperature: ROOM_TEMPERATURE_C,
            reference_pressure: ATMOSPHERIC_PRESSURE,
        }
    }

    /// Validate material properties against physical constraints
    /// # Errors
    /// - Returns [`KwaversError::Medium`] if the precondition for a Medium-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        use kwavers_core::error::MediumError;

        // Speed of sound must be positive
        if self.sound_speed <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "sound_speed".to_owned(),
                value: self.sound_speed,
                constraint: "must be positive".to_owned(),
            }));
        }

        // Density must be positive
        if self.density <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "density".to_owned(),
                value: self.density,
                constraint: "must be positive".to_owned(),
            }));
        }

        // Impedance must be positive
        if self.impedance <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "impedance".to_owned(),
                value: self.impedance,
                constraint: "must be positive".to_owned(),
            }));
        }

        // Absorption must be non-negative
        if self.absorption_coefficient < 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "absorption_coefficient".to_owned(),
                value: self.absorption_coefficient,
                constraint: "must be non-negative".to_owned(),
            }));
        }

        // Specific heat must be positive
        if self.specific_heat <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "specific_heat".to_owned(),
                value: self.specific_heat,
                constraint: "must be positive".to_owned(),
            }));
        }

        // Thermal conductivity must be non-negative
        if self.thermal_conductivity < 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "thermal_conductivity".to_owned(),
                value: self.thermal_conductivity,
                constraint: "must be non-negative".to_owned(),
            }));
        }

        // Viscosity must be non-negative
        if self.shear_viscosity < 0.0 || self.bulk_viscosity < 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "viscosity".to_owned(),
                value: self.shear_viscosity.min(self.bulk_viscosity),
                constraint: "must be non-negative".to_owned(),
            }));
        }

        // Refractive index should be typically >= 1.0 for physical materials
        if self.refractive_index < 1.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "refractive_index".to_owned(),
                value: self.refractive_index,
                constraint: "must be >= 1.0".to_owned(),
            }));
        }

        Ok(())
    }

    /// Calculate acoustic impedance mismatch ratio (|reflection coefficient|).
    #[must_use]
    pub fn impedance_ratio(&self, other: &Self) -> f64 {
        reflection_coefficient(self.impedance, other.impedance).abs()
    }

    /// Calculate reflection coefficient magnitude at normal incidence.
    #[must_use]
    pub fn reflection_coefficient(&self, other: &Self) -> f64 {
        reflection_coefficient(self.impedance, other.impedance).abs()
    }

    /// Pressure-amplitude transmission coefficient `T = 2 Z_other/(Z_self + Z_other)`
    /// at normal incidence (the pressure companion of [`reflection_coefficient`],
    /// satisfying `T = 1 + R` with the *signed* `R`).
    ///
    /// [`reflection_coefficient`]: Self::reflection_coefficient
    #[must_use]
    pub fn transmission_coefficient(&self, other: &Self) -> f64 {
        transmission_coefficient(self.impedance, other.impedance)
    }

    /// Absorption coefficient at given frequency
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        self.absorption_coefficient * frequency.powf(self.absorption_exponent)
    }

    /// Attenuation in dB/cm at given frequency
    #[must_use]
    pub fn attenuation_db_cm(&self, frequency: f64) -> f64 {
        let alpha = self.absorption_at_frequency(frequency);
        // Convert Np/m to dB/cm: dB = 20·log₁₀(e) · Np · 0.01
        alpha * 20.0 * std::f64::consts::LOG10_E * 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use kwavers_core::constants::thermodynamic::SPECIFIC_HEAT_WATER;

    #[test]
    fn test_material_creation() {
        let mat = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        assert_eq!(mat.sound_speed, SOUND_SPEED_WATER_SIM);
        assert_eq!(mat.impedance, SOUND_SPEED_WATER_SIM * DENSITY_WATER_NOMINAL);
    }

    #[test]
    fn test_validation_valid() {
        let mat = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        mat.validate().unwrap();
    }

    #[test]
    fn test_validation_invalid_speed() {
        let mut mat = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        mat.sound_speed = -SOUND_SPEED_WATER_SIM;
        assert!(mat.validate().is_err());
    }

    #[test]
    fn test_impedance_match() {
        let mat = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        let same = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        assert!((mat.reflection_coefficient(&same)).abs() < 1e-6);
    }

    #[test]
    fn transmission_coefficient_is_pressure_amplitude() {
        // Water (Z≈1.5 MRayl) → bone-like (Z≈5.7 MRayl): Z_t > Z_i ⇒ R > 0 and
        // the pressure transmission T = 1 + R EXCEEDS 1 (the case the old
        // `1 − |R|` form got wrong).
        let water = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        let bone = AcousticMaterialProperties::new(3000.0, 1900.0, 0.002, SPECIFIC_HEAT_WATER, 0.6);
        let (zi, zt) = (water.impedance, bone.impedance);

        let t = water.transmission_coefficient(&bone);
        // Closed form 2 Z_t/(Z_i+Z_t).
        assert!((t - 2.0 * zt / (zi + zt)).abs() < 1e-9);
        // Pressure transmission exceeds unity into the stiffer medium.
        assert!(t > 1.0, "T into higher impedance must exceed 1, got {t}");
        // Relation T = 1 + R (signed reflection).
        let r = super::reflection_coefficient(zi, zt);
        assert!((t - (1.0 + r)).abs() < 1e-12);
        // Lossless energy balance R² + (Z_i/Z_t)·T² = 1.
        assert!((r.mul_add(r, (zi / zt) * t * t) - 1.0).abs() < 1e-9);

        // Matched impedance ⇒ R = 0, T = 1.
        assert!((water.transmission_coefficient(&water) - 1.0).abs() < 1e-12);
    }

    /// Corollary 1.2: R_I = R², T_I = 4 Z_i Z_t/(Z_i+Z_t)², and the lossless
    /// energy balance R_I + T_I = 1.
    #[test]
    fn intensity_coefficients_conserve_energy() {
        // Water → bone-like.
        let (zi, zt) = (1.5e6, 5.7e6);
        let r_i = intensity_reflection_coefficient(zi, zt);
        let t_i = intensity_transmission_coefficient(zi, zt);
        let r = reflection_coefficient(zi, zt);
        assert!((r_i - r * r).abs() < 1e-12, "R_I must equal R²");
        assert!(
            (t_i - 4.0 * zi * zt / ((zi + zt) * (zi + zt))).abs() < 1e-9,
            "T_I closed form"
        );
        assert!((r_i + t_i - 1.0).abs() < 1e-12, "R_I + T_I must equal 1");
        // Matched impedance ⇒ no reflection, full transmission.
        assert!(intensity_reflection_coefficient(zi, zi).abs() < 1e-12);
        assert!((intensity_transmission_coefficient(zi, zi) - 1.0).abs() < 1e-12);
        // Degenerate (zero impedance sum) guard.
        assert_eq!(intensity_transmission_coefficient(0.0, 0.0), 0.0);
    }

    /// Theorem 1.6: I = A²/(2Z) = p_rms²/Z for a harmonic plane wave.
    #[test]
    fn plane_wave_intensity_matches_theorem() {
        let z = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM; // rayl
        let amp = 1.0e6; // 1 MPa
        let i = plane_wave_intensity(amp, z);
        assert!((i - amp * amp / (2.0 * z)).abs() < 1e-6);
        // Equivalent p_rms form: p_rms = A/√2 ⇒ I = p_rms²/Z.
        let p_rms = amp / 2.0_f64.sqrt();
        assert!((i - p_rms * p_rms / z).abs() < 1e-3);
        // Non-positive impedance ⇒ 0.
        assert_eq!(plane_wave_intensity(amp, 0.0), 0.0);
    }

    /// Definition 3.2: β = 1 + B/(2A). Water B/A = 5 ⇒ β = 3.5; B/A = 0 ⇒ β = 1.
    #[test]
    fn coefficient_of_nonlinearity_matches_definition() {
        assert!((coefficient_of_nonlinearity(5.0) - 3.5).abs() < 1e-12);
        assert!((coefficient_of_nonlinearity(0.0) - 1.0).abs() < 1e-12);
        assert!((coefficient_of_nonlinearity(9.6) - 5.8).abs() < 1e-12); // fat
    }

    #[test]
    fn test_attenuation_frequency_dependence() {
        let mat = AcousticMaterialProperties::new(
            SOUND_SPEED_WATER_SIM,
            DENSITY_WATER_NOMINAL,
            0.002,
            SPECIFIC_HEAT_WATER,
            0.6,
        );
        let att_1mhz = mat.absorption_at_frequency(MHZ_TO_HZ);
        let att_2mhz = mat.absorption_at_frequency(2.0 * MHZ_TO_HZ);
        // With exponent 1.0, doubling frequency doubles absorption
        assert!(att_2mhz > att_1mhz);
    }
}
