//! Photoacoustic Physics Implementations
//!
//! This module implements photoacoustic coupling physics, including
//! optical absorption, thermal expansion, and pressure wave generation.

/// Grüneisen parameter for thermoelastic coupling
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

    /// Get Grüneisen parameter value (could depend on conditions)
    pub fn get_value(&self, _temperature: f64, _pressure: f64) -> f64 {
        // For now, return constant value
        // Full implementation would include temperature/pressure dependence
        self.value
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

    #[test]
    fn test_gruneisen_parameter() {
        let gamma = GruneisenParameter::new(0.5);
        assert_eq!(gamma.get_value(310.0, 1e5), 0.5);
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
