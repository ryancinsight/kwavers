use kwavers_core::constants::acoustic_parameters::{
    BONE_DENSITY, DENSITY_SKULL_SUTURE, DENSITY_SKULL_TRABECULAR, SHEAR_SPEED_SKULL_CORTICAL,
    SHEAR_SPEED_SKULL_TRABECULAR, SOUND_SPEED_SKULL_CORTICAL, SOUND_SPEED_SKULL_SUTURE,
    SOUND_SPEED_SKULL_TRABECULAR,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Skull material properties based on literature
///
/// Reference: Pinton et al. (2012) "Attenuation, scattering, and absorption
/// of ultrasound in the skull bone"
#[derive(Debug, Clone)]
pub struct AcousticSkullProperties {
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

impl Default for AcousticSkullProperties {
    fn default() -> Self {
        // Typical adult skull properties — Pinton et al. (2012)
        Self {
            sound_speed: SOUND_SPEED_SKULL_CORTICAL, // 3100.0 m/s — Pinton (2012)
            density: BONE_DENSITY,                   // 1900 kg/m³ — Duck (1990)
            attenuation_coeff: 60.0,                 // Np/m/MHz
            thickness: 0.007,                        // 7 mm average
            shear_speed: Some(SHEAR_SPEED_SKULL_CORTICAL), // 1600.0 m/s — Pinton (2012)
        }
    }
}

impl AcousticSkullProperties {
    /// Create skull properties for specific bone types
    ///
    /// # Arguments
    ///
    /// * `bone_type` - "cortical", "trabecular", or "suture"
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn from_bone_type(bone_type: &str) -> KwaversResult<Self> {
        match bone_type {
            "cortical" => Ok(Self {
                sound_speed: SOUND_SPEED_SKULL_CORTICAL, // 3100.0 m/s — Pinton (2012)
                density: BONE_DENSITY,                   // 1900.0 kg/m³ — Duck (1990)
                attenuation_coeff: 60.0,
                thickness: 0.007,
                shear_speed: Some(SHEAR_SPEED_SKULL_CORTICAL), // 1600.0 m/s — Pinton (2012)
            }),
            "trabecular" => Ok(Self {
                sound_speed: SOUND_SPEED_SKULL_TRABECULAR, // 2400.0 m/s — Pinton (2012)
                density: DENSITY_SKULL_TRABECULAR,         // 1600.0 kg/m³ — Pinton (2012)
                attenuation_coeff: 40.0,
                thickness: 0.005,
                shear_speed: Some(SHEAR_SPEED_SKULL_TRABECULAR), // 1200.0 m/s — Pinton (2012)
            }),
            "suture" => Ok(Self {
                sound_speed: SOUND_SPEED_SKULL_SUTURE, // 1800.0 m/s — Pinton (2012)
                density: DENSITY_SKULL_SUTURE,         // 1200.0 kg/m³ — Pinton (2012)
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
    #[must_use]
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.sound_speed
    }

    /// Calculate intensity transmission coefficient for normal incidence (water → skull)
    ///
    /// For a plane wave at normal incidence from medium 1 (water, Z₁) to medium 2
    /// (skull, Z₂) the intensity transmission coefficient is (Kinsler et al. 2000 §6.2):
    ///
    /// ```text
    /// T_I = 4 Z₁ Z₂ / (Z₁ + Z₂)²
    /// ```
    ///
    /// This is the fraction of incident acoustic intensity transmitted into the skull.
    /// Note: the pressure amplitude TC is T_p = 2Z₂/(Z₁+Z₂), which can exceed 1 when
    /// Z₂ > Z₁ (pressure doubling at hard boundary); T_I is always ≤ 1.
    ///
    /// Reference: Kinsler et al. (2000) "Fundamentals of Acoustics" §6.2
    #[must_use]
    pub fn transmission_coefficient(&self, water_impedance: f64) -> f64 {
        let z_skull = self.acoustic_impedance();
        // T_I = 4 Z1 Z2 / (Z1 + Z2)^2
        4.0 * water_impedance * z_skull / (water_impedance + z_skull).powi(2)
    }

    /// Calculate attenuation for given frequency (Hz)
    ///
    /// Attenuation = α(f) = α₀ × f^n where n ≈ 1 for bone
    #[must_use]
    pub fn attenuation_at_frequency(&self, frequency: f64) -> f64 {
        let freq_mhz = frequency / MHZ_TO_HZ;
        self.attenuation_coeff * freq_mhz
    }
}
