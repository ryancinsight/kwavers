//! Photoacoustic Physics Implementations
//!
//! This module implements photoacoustic coupling physics, including
//! optical absorption, thermal expansion, and pressure wave generation.

use crate::core::constants::thermodynamic::{BODY_TEMPERATURE_C, GRUNEISEN_WATER_37C, P_ATM};

/// Grüneisen parameter for thermoelastic coupling
/// TODO_AUDIT: P2 - Advanced Photoacoustic Physics - Implement complete photoacoustic wave generation with nonlinear thermoelasticity and acoustic saturation
/// DEPENDS ON: physics/electromagnetic/photoacoustic/nonlinear_thermoelastic.rs, physics/electromagnetic/photoacoustic/saturation.rs, physics/electromagnetic/photoacoustic/broadband.rs
/// MISSING: Nonlinear Grüneisen parameter: Γ(T,p) = Γ₀ (1 + α ΔT + β Δp) with temperature/pressure dependence
/// MISSING: Acoustic saturation effects at high laser fluences (>100 mJ/cm²)
/// MISSING: Broadband photoacoustic signals with frequency-dependent absorption
/// MISSING: Stress confinement vs thermal confinement regime analysis
/// MISSING: Multi-wavelength photoacoustic spectroscopy with chemical specificity
/// SEVERITY: HIGH (enables quantitative photoacoustic imaging)
/// THEOREM: Photoacoustic pressure: p₀ = Γ μₐ Φ / C_p where Γ = β c² / (C_p ρ) is Grüneisen parameter
/// THEOREM: Thermal confinement: μₐ δ_th << 1 where δ_th = √(κ t_p) is thermal diffusion length
/// REFERENCES: Oraevsky & Karabutov (2003) Biomedical Photonics; Wang & Wu (2007) Biomedical Optics
#[derive(Debug, Clone)]
pub struct GruneisenParameter {
    /// Grüneisen parameter Γ (dimensionless)
    pub value: f64,
    /// Temperature dependence (dΓ/dT)
    pub temperature_coefficient: Option<f64>,
    /// Pressure dependence (dΓ/dP)
    pub pressure_coefficient: Option<f64>,
}

impl GruneisenParameter {
    /// Create a new Grüneisen parameter
    pub fn new(value: f64) -> Self {
        Self {
            value,
            temperature_coefficient: None,
            pressure_coefficient: None,
        }
    }

    /// Create with temperature dependence
    pub fn with_temperature_dependence(mut self, coeff: f64) -> Self {
        self.temperature_coefficient = Some(coeff);
        self
    }

    /// Create with pressure dependence
    pub fn with_pressure_dependence(mut self, coeff: f64) -> Self {
        self.pressure_coefficient = Some(coeff);
        self
    }

    /// Evaluate the Grüneisen parameter at the given temperature and pressure.
    ///
    /// ## Theorem (Sigrist & Kneubühl 1978, Eq. 4; Wang & Wu 2012, §2.3)
    ///
    /// Near a reference state (T₀, p₀) the Grüneisen parameter varies linearly:
    ///
    /// ```text
    /// Γ(T, p) = Γ₀ · [1 + α·(T − T₀) + β·(p − p₀)]
    /// ```
    ///
    /// where:
    /// - `T₀ = BODY_TEMPERATURE_C` (37 °C) — reference temperature
    /// - `p₀ = P_ATM` (101 325 Pa) — reference pressure
    /// - `α` = temperature coefficient \[K⁻¹\], `None` → 0 (constant Γ)
    /// - `β` = pressure coefficient \[Pa⁻¹\], `None` → 0 (constant Γ)
    ///
    /// When both coefficients are `None` the function degenerates to the
    /// constant model `Γ(T, p) = Γ₀`, which is the correct result for a
    /// temperature- and pressure-independent medium (e.g. homogeneous phantom).
    ///
    /// ## References
    /// - Sigrist, M.W. & Kneubühl, F.K. (1978). "Laser-generated stress waves in
    ///   solids." J. Acoust. Soc. Am. 64(6), 1652–1663. DOI: 10.1121/1.382132
    /// - Wang, L.V. & Wu, H.-I. (2012). *Biomedical Optics: Principles and Imaging*,
    ///   §2.3. Wiley-Interscience. ISBN 978-0-471-74304-0.
    pub fn get_value(&self, temperature: f64, pressure: f64) -> f64 {
        let dt = temperature - BODY_TEMPERATURE_C; // deviation from 37 °C
        let dp = pressure - P_ATM; // deviation from 101 325 Pa
        self.value
            * (1.0
                + self.temperature_coefficient.unwrap_or(0.0) * dt
                + self.pressure_coefficient.unwrap_or(0.0) * dp)
    }
}

/// Optical absorption properties
#[derive(Debug, Clone)]
pub struct OpticalAbsorption {
    /// Absorption coefficient μ_a (m⁻¹)
    pub absorption_coefficient: f64,
    /// Reduced scattering coefficient μ_s' (m⁻¹)
    pub reduced_scattering: f64,
    /// Anisotropy factor g (dimensionless, -1 to 1)
    pub anisotropy_factor: f64,
    /// Optical wavelength (m)
    pub wavelength: f64,
}

impl OpticalAbsorption {
    /// Create optical absorption properties
    pub fn new(mu_a: f64, mu_s_prime: f64, g: f64, wavelength: f64) -> Self {
        Self {
            absorption_coefficient: mu_a,
            reduced_scattering: mu_s_prime,
            anisotropy_factor: g,
            wavelength,
        }
    }

    /// Get total attenuation coefficient μ_t = μ_a + μ_s'
    pub fn total_attenuation(&self) -> f64 {
        self.absorption_coefficient + self.reduced_scattering
    }

    /// Get optical penetration depth δ = 1/μ_t
    pub fn penetration_depth(&self) -> f64 {
        1.0 / self.total_attenuation()
    }

    /// Get albedo (scattering probability) a = μ_s' / μ_t
    pub fn albedo(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 {
            self.reduced_scattering / mu_t
        } else {
            0.0
        }
    }
}

/// Tissue optical properties database
#[derive(Debug)]
pub struct TissueOpticalProperties;

impl TissueOpticalProperties {
    /// Get optical properties for a specific tissue type at wavelength
    pub fn get_properties(tissue_type: &str, wavelength: f64) -> Option<OpticalAbsorption> {
        match tissue_type {
            "blood" => Some(OpticalAbsorption::new(
                200.0, // μ_a ≈ 200 cm⁻¹ at 800 nm
                100.0, // μ_s' ≈ 100 cm⁻¹
                0.99,  // g ≈ 0.99 (highly forward scattering)
                wavelength,
            )),
            "muscle" => Some(OpticalAbsorption::new(
                5.0,  // μ_a ≈ 5 cm⁻¹
                50.0, // μ_s' ≈ 50 cm⁻¹
                0.9,  // g ≈ 0.9
                wavelength,
            )),
            "fat" => Some(OpticalAbsorption::new(
                2.0,  // μ_a ≈ 2 cm⁻¹
                30.0, // μ_s' ≈ 30 cm⁻¹
                0.8,  // g ≈ 0.8
                wavelength,
            )),
            "skin" => Some(OpticalAbsorption::new(
                20.0,  // μ_a ≈ 20 cm⁻¹
                150.0, // μ_s' ≈ 150 cm⁻¹
                0.8,   // g ≈ 0.8
                wavelength,
            )),
            _ => None,
        }
    }
}

/// Pulsed laser source for photoacoustic excitation
#[derive(Debug)]
pub struct PulsedLaser {
    /// Peak power (W)
    pub peak_power: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Repetition rate (Hz)
    pub repetition_rate: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Beam profile
    pub beam_profile: BeamProfile,
}

#[derive(Debug, Clone)]
pub enum BeamProfile {
    Gaussian { beam_radius: f64 },
    FlatTop { beam_radius: f64 },
    Bessel { central_lobe_radius: f64 },
}

impl PulsedLaser {
    /// Create a new pulsed laser
    pub fn new(peak_power: f64, pulse_duration: f64, wavelength: f64) -> Self {
        Self {
            peak_power,
            pulse_duration,
            repetition_rate: 10.0, // 10 Hz default
            wavelength,
            beam_profile: BeamProfile::Gaussian { beam_radius: 1e-3 },
        }
    }

    /// Compute peak fluence (J/m²)
    pub fn peak_fluence(&self) -> f64 {
        match &self.beam_profile {
            BeamProfile::Gaussian { beam_radius } => {
                // For Gaussian beam: Φ₀ = (2E_pulse)/(π w₀²)
                let beam_area = std::f64::consts::PI * beam_radius * beam_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                2.0 * pulse_energy / beam_area
            }
            BeamProfile::FlatTop { beam_radius } => {
                let beam_area = std::f64::consts::PI * beam_radius * beam_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                pulse_energy / beam_area
            }
            BeamProfile::Bessel {
                central_lobe_radius,
            } => {
                // Simplified for central lobe
                let beam_area = std::f64::consts::PI * central_lobe_radius * central_lobe_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                pulse_energy / beam_area
            }
        }
    }

    /// Compute average power (W)
    pub fn average_power(&self) -> f64 {
        self.peak_power * self.pulse_duration * self.repetition_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without coefficients the model reduces to the constant Γ₀ regardless of T, p.
    #[test]
    fn test_gruneisen_constant_no_coefficients() {
        let gamma = GruneisenParameter::new(0.12);
        // Any T and p should return Γ₀ when coefficients are None
        let v1 = gamma.get_value(BODY_TEMPERATURE_C, P_ATM);
        let v2 = gamma.get_value(20.0, 0.5e5);
        let v3 = gamma.get_value(80.0, 5e5);
        assert!((v1 - 0.12).abs() < 1e-15);
        assert!((v2 - 0.12).abs() < 1e-15);
        assert!((v3 - 0.12).abs() < 1e-15);
    }

    /// Γ(T_ref, p_ref) == Γ₀ even when coefficients are set.
    #[test]
    fn test_gruneisen_reference_conditions_unchanged() {
        let gamma = GruneisenParameter::new(0.12)
            .with_temperature_dependence(0.005)
            .with_pressure_dependence(1e-6);
        let v = gamma.get_value(BODY_TEMPERATURE_C, P_ATM);
        assert!(
            (v - 0.12).abs() < 1e-15,
            "At reference conditions Γ must equal Γ₀, got {v}"
        );
    }

    /// α > 0 → Γ increases with temperature.
    #[test]
    fn test_gruneisen_temperature_linear() {
        // Γ₀ = 0.12, α = 0.01 K⁻¹, T = T_ref + 10 K → Γ = 0.12 * 1.1 = 0.132
        let gamma = GruneisenParameter::new(0.12).with_temperature_dependence(0.01);
        let v = gamma.get_value(BODY_TEMPERATURE_C + 10.0, P_ATM);
        let expected = 0.12 * 1.1;
        assert!(
            (v - expected).abs() < 1e-12,
            "Expected {expected}, got {v}"
        );
    }

    /// β > 0 → Γ increases with pressure.
    #[test]
    fn test_gruneisen_pressure_linear() {
        // Γ₀ = 0.12, β = 1e-6 Pa⁻¹, p = p_ref + 1e5 Pa → Γ = 0.12 * 1.1 = 0.132
        let gamma = GruneisenParameter::new(0.12).with_pressure_dependence(1e-6);
        let v = gamma.get_value(BODY_TEMPERATURE_C, P_ATM + 1e5);
        let expected = 0.12 * 1.1;
        assert!(
            (v - expected).abs() < 1e-12,
            "Expected {expected}, got {v}"
        );
    }

    #[test]
    fn test_optical_absorption() {
        let absorption = OpticalAbsorption::new(10.0, 50.0, 0.9, 800e-9);
        assert_eq!(absorption.total_attenuation(), 60.0);
        assert_eq!(absorption.albedo(), 50.0 / 60.0);
    }

    #[test]
    fn test_tissue_optical_properties() {
        let blood_props = TissueOpticalProperties::get_properties("blood", 800e-9);
        assert!(blood_props.is_some());

        let blood = blood_props.unwrap();
        assert!(blood.absorption_coefficient > 0.0);
        assert!(blood.reduced_scattering > 0.0);
    }

    #[test]
    fn test_pulsed_laser() {
        let laser = PulsedLaser::new(1e6, 10e-9, 800e-9); // 1 MW peak, 10 ns pulse

        let peak_fluence = laser.peak_fluence();
        assert!(peak_fluence > 0.0);

        let avg_power = laser.average_power();
        assert!(avg_power < laser.peak_power); // Average should be less than peak
    }
}
