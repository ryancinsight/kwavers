//! Photoacoustic Physics Implementations
//!
//! This module implements photoacoustic coupling physics, including
//! optical absorption, thermal expansion, and pressure wave generation.

use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;

/// Canonical Grüneisen model — temperature-dependent, literature-validated.
///
/// This is the single source of truth for Grüneisen parameter evaluation in kwavers
/// (Sprint 226 SSOT consolidation). Use this type everywhere a Grüneisen value is
/// needed; `GruneisenParameter` is deprecated and will be removed in a future sprint.
pub use crate::photoacoustics::thermoelasticity::GrueneisenModel;

/// Grüneisen parameter for thermoelastic coupling.
///
/// # Deprecated
///
/// Use [`GrueneisenModel`] instead. `GrueneisenModel` is the canonical, SSOT type with
/// correct temperature dependence and literature-validated constants (Sigrist 1986;
/// Xu & Wang 2006). This struct adds a pressure-dependence term (β ≈ 10⁻⁶ Pa⁻¹)
/// that is negligible over physiological pressures (Wang & Wu 2012, §2.3) and
/// diverges from the canonical water reference temperature (20 °C vs 37 °C here).
#[deprecated(
    since = "0.4.0",
    note = "Use `GrueneisenModel` from `physics::photoacoustics::thermoelasticity` instead"
)]
///
/// ## Not yet implemented
///
/// - **Nonlinear Grüneisen parameter**: Γ(T,p) = Γ₀(1 + αΔT + βΔp) with full
///   temperature and pressure dependence (Wang & Wu 2007, Biomedical Optics).
/// - **Acoustic saturation**: Pressure limiting at high laser fluences above 100 mJ/cm².
/// - **Broadband signals**: Frequency-dependent optical absorption for wideband transducers.
/// - **Confinement regime analysis**: Stress vs. thermal confinement transition logic.
/// - **Multi-wavelength spectroscopy**: Wavelength-selective imaging for chemical specificity
///   (Oraevsky & Karabutov 2003, Biomedical Photonics).
#[derive(Debug, Clone)]
pub struct GruneisenParameter {
    /// Grüneisen parameter Γ (dimensionless)
    pub value: f64,
    /// Temperature dependence (dΓ/dT)
    pub temperature_coefficient: Option<f64>,
    /// Pressure dependence (dΓ/dP)
    pub pressure_coefficient: Option<f64>,
}

#[allow(deprecated)]
impl GruneisenParameter {
    /// Create a new Grüneisen parameter
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value,
            temperature_coefficient: None,
            pressure_coefficient: None,
        }
    }

    /// Create with temperature dependence
    #[must_use]
    pub fn with_temperature_dependence(mut self, coeff: f64) -> Self {
        self.temperature_coefficient = Some(coeff);
        self
    }

    /// Create with pressure dependence
    #[must_use]
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
    /// - `p₀ = ATMOSPHERIC_PRESSURE` (101 325 Pa) — reference pressure
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
    #[must_use]
    pub fn get_value(&self, temperature: f64, pressure: f64) -> f64 {
        let dt = temperature - BODY_TEMPERATURE_C; // deviation from 37 °C
        let dp = pressure - ATMOSPHERIC_PRESSURE; // deviation from 101 325 Pa
        self.value
            * self.pressure_coefficient.unwrap_or(0.0).mul_add(
                dp,
                self.temperature_coefficient.unwrap_or(0.0).mul_add(dt, 1.0),
            )
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
    #[must_use]
    pub fn new(mu_a: f64, mu_s_prime: f64, g: f64, wavelength: f64) -> Self {
        Self {
            absorption_coefficient: mu_a,
            reduced_scattering: mu_s_prime,
            anisotropy_factor: g,
            wavelength,
        }
    }

    /// Get total attenuation coefficient μ_t = μ_a + μ_s'
    #[must_use]
    pub fn total_attenuation(&self) -> f64 {
        self.absorption_coefficient + self.reduced_scattering
    }

    /// Get optical penetration depth δ = 1/μ_t
    #[must_use]
    pub fn penetration_depth(&self) -> f64 {
        1.0 / self.total_attenuation()
    }

    /// Get albedo (scattering probability) a = μ_s' / μ_t
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn average_power(&self) -> f64 {
        self.peak_power * self.pulse_duration * self.repetition_rate
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use kwavers_core::constants::thermodynamic::GRUNEISEN_WATER_20C;

    /// Without coefficients the model reduces to the constant Γ₀ regardless of T, p.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_gruneisen_constant_no_coefficients() {
        let gamma = GruneisenParameter::new(GRUNEISEN_WATER_20C);
        // Any T and p should return Γ₀ when coefficients are None
        let v1 = gamma.get_value(BODY_TEMPERATURE_C, ATMOSPHERIC_PRESSURE);
        let v2 = gamma.get_value(20.0, 0.5e5);
        let v3 = gamma.get_value(80.0, 5e5);
        assert!((v1 - GRUNEISEN_WATER_20C).abs() < 1e-15);
        assert!((v2 - GRUNEISEN_WATER_20C).abs() < 1e-15);
        assert!((v3 - GRUNEISEN_WATER_20C).abs() < 1e-15);
    }

    /// Γ(T_ref, p_ref) == Γ₀ even when coefficients are set.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_gruneisen_reference_conditions_unchanged() {
        let gamma = GruneisenParameter::new(GRUNEISEN_WATER_20C)
            .with_temperature_dependence(0.005)
            .with_pressure_dependence(1e-6);
        let v = gamma.get_value(BODY_TEMPERATURE_C, ATMOSPHERIC_PRESSURE);
        assert!(
            (v - GRUNEISEN_WATER_20C).abs() < 1e-15,
            "At reference conditions Γ must equal Γ₀, got {v}"
        );
    }

    /// α > 0 → Γ increases with temperature.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_gruneisen_temperature_linear() {
        // Γ₀ = GRUNEISEN_WATER_20C (0.12), α = 0.01 K⁻¹, T = T_ref + 10 K → Γ = Γ₀ * 1.1 = 0.132
        let gamma = GruneisenParameter::new(GRUNEISEN_WATER_20C).with_temperature_dependence(0.01);
        let v = gamma.get_value(BODY_TEMPERATURE_C + 10.0, ATMOSPHERIC_PRESSURE);
        let expected = GRUNEISEN_WATER_20C * 1.1;
        assert!((v - expected).abs() < 1e-12, "Expected {expected}, got {v}");
    }

    /// β > 0 → Γ increases with pressure.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_gruneisen_pressure_linear() {
        // Γ₀ = GRUNEISEN_WATER_20C (0.12), β = 1e-6 Pa⁻¹, p = p_ref + 1e5 Pa → Γ = Γ₀ * 1.1 = 0.132
        let gamma = GruneisenParameter::new(GRUNEISEN_WATER_20C).with_pressure_dependence(1e-6);
        let v = gamma.get_value(BODY_TEMPERATURE_C, ATMOSPHERIC_PRESSURE + 1e5);
        let expected = GRUNEISEN_WATER_20C * 1.1;
        assert!((v - expected).abs() < 1e-12, "Expected {expected}, got {v}");
    }

    #[test]
    fn test_optical_absorption() {
        let absorption = OpticalAbsorption::new(10.0, 50.0, 0.9, 800e-9);
        assert_eq!(absorption.total_attenuation(), 60.0);
        assert_eq!(absorption.albedo(), 50.0 / 60.0);
    }

    #[test]
    fn test_tissue_optical_properties() {
        let blood = TissueOpticalProperties::get_properties("blood", 800e-9).unwrap();
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
