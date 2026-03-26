use crate::core::error::{KwaversError, KwaversResult};

/// Skull material properties based on literature
///
/// Reference: Pinton et al. (2012) "Attenuation, scattering, and absorption
/// of ultrasound in the skull bone"
#[derive(Debug, Clone)]
pub struct SkullProperties {
    /// Sound speed in skull (m/s) - typically 2800-3500 m/s
    pub sound_speed: f64,
    /// Density (kg/m³) - typically 1850-2000 kg/m³
    pub density: f64,
    /// Attenuation coefficient (Np/m/MHz) - typically 40-100
    pub attenuation_coeff: f64,
    /// Skull thickness (m) - typically 3-10 mm
    pub thickness: f64,
    /// Shear wave speed (m/s) - typically 1400-1800 m/s
    pub shear_speed: Option<f64>,
}

impl Default for SkullProperties {
    fn default() -> Self {
        // Typical adult skull properties
        Self {
            sound_speed: 3100.0,       // m/s (cortical bone)
            density: 1900.0,           // kg/m³
            attenuation_coeff: 60.0,   // Np/m/MHz
            thickness: 0.007,          // 7 mm average
            shear_speed: Some(1600.0), // m/s
        }
    }
}

impl SkullProperties {
    /// Create skull properties for specific bone types
    ///
    /// # Arguments
    ///
    /// * `bone_type` - "cortical", "trabecular", or "suture"
    pub fn from_bone_type(bone_type: &str) -> KwaversResult<Self> {
        match bone_type {
            "cortical" => Ok(Self {
                sound_speed: 3100.0,
                density: 1900.0,
                attenuation_coeff: 60.0,
                thickness: 0.007,
                shear_speed: Some(1600.0),
            }),
            "trabecular" => Ok(Self {
                sound_speed: 2400.0,
                density: 1600.0,
                attenuation_coeff: 40.0,
                thickness: 0.005,
                shear_speed: Some(1200.0),
            }),
            "suture" => Ok(Self {
                sound_speed: 1800.0,
                density: 1200.0,
                attenuation_coeff: 20.0,
                thickness: 0.002,
                shear_speed: None, // Soft tissue-like
            }),
            _ => Err(KwaversError::InvalidInput(format!(
                "Unknown bone type: {}",
                bone_type
            ))),
        }
    }

    /// Calculate acoustic impedance (Z = ρc)
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.sound_speed
    }

    /// Calculate transmission coefficient for normal incidence
    ///
    /// Reference: Kinsler et al. (2000) "Fundamentals of Acoustics"
    pub fn transmission_coefficient(&self, water_impedance: f64) -> f64 {
        let z_skull = self.acoustic_impedance();
        let t = 2.0 * water_impedance / (water_impedance + z_skull);
        t.powi(2) // Intensity transmission coefficient
    }

    /// Calculate attenuation for given frequency (Hz)
    ///
    /// Attenuation = α(f) = α₀ × f^n where n ≈ 1 for bone
    pub fn attenuation_at_frequency(&self, frequency: f64) -> f64 {
        let freq_mhz = frequency / 1e6;
        self.attenuation_coeff * freq_mhz
    }
}
