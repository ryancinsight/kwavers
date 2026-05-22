//! Acoustic material property data structures
//!
//! # Mathematical Foundation
//!
//! Wave equation with absorption and nonlinearity:
//! ```text
//! ∂²p/∂t² = c²∇²p - 2α(∂p/∂t) + (β/ρc²)(∇p)²
//! ```
//!
//! Where:
//! - `p`: Acoustic pressure (Pa)
//! - `c`: Sound speed (m/s)
//! - `ρ`: Density (kg/m³)
//! - `α(f)`: Frequency-dependent absorption coefficient (Np/m)
//! - `β`: Nonlinearity coefficient (dimensionless)
//!
//! ## Absorption Model
//!
//! Power-law frequency dependence:
//! ```text
//! α(f) = α₀ · f^y
//! ```
//! - `α₀`: Absorption coefficient (Np/(MHz^y m))
//! - `f`: Frequency (MHz)
//! - `y`: Power exponent (typical: 1.0-2.0)
//!
//! ## Impedance
//!
//! Acoustic impedance:
//! ```text
//! Z = ρc  (kg/m²s or Rayl)
//! ```
//!
//! ## Invariants
//!
//! - `density > 0`
//! - `sound_speed > 0`
//! - `absorption_coefficient ≥ 0`
//! - `0.5 ≤ absorption_power ≤ 3.0` (physical range)
//! - `nonlinearity > 0` (typically 3-10 for biological media)

use crate::core::constants::fundamental::{
    B_OVER_A_BRAIN, B_OVER_A_FAT, B_OVER_A_KIDNEY, B_OVER_A_LIVER, B_OVER_A_MUSCLE,
    B_OVER_A_WATER, DENSITY_BRAIN, DENSITY_FAT, DENSITY_LIVER, DENSITY_MUSCLE, DENSITY_TISSUE,
    DENSITY_WATER, SOUND_SPEED_BRAIN, SOUND_SPEED_FAT, SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER,
    SOUND_SPEED_MUSCLE, SOUND_SPEED_TISSUE,
};
use std::fmt;

/// Canonical acoustic material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticPropertyData {
    /// Density ρ (kg/m³)
    ///
    /// Physical range: 1-20000 kg/m³ (air to dense metals)
    pub density: f64,

    /// Sound speed c (m/s)
    ///
    /// Physical range: 300-6000 m/s (air to solids)
    pub sound_speed: f64,

    /// Absorption coefficient α₀ (Np/(MHz^y m))
    ///
    /// Power-law prefactor for frequency-dependent absorption.
    /// For water: ~0.002 Np/(MHz² m) with y=2
    pub absorption_coefficient: f64,

    /// Absorption power exponent y (dimensionless)
    ///
    /// Physical range: 0.5-3.0
    /// - Water: y ≈ 2.0
    /// - Soft tissue: y ≈ 1.0-1.5
    /// - Air: y ≈ 2.0
    pub absorption_power: f64,

    /// Nonlinearity parameter B/A (dimensionless)
    ///
    /// Physical range: 3-10 for biological media
    /// - Water: B/A ≈ 5.0
    /// - Tissue: B/A ≈ 6.0-8.0
    pub nonlinearity: f64,
}

impl AcousticPropertyData {
    /// Construct with validation of physical constraints
    ///
    /// # Errors
    ///
    /// Returns error message if any parameter violates physical bounds
    pub fn new(
        density: f64,
        sound_speed: f64,
        absorption_coefficient: f64,
        absorption_power: f64,
        nonlinearity: f64,
    ) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if sound_speed <= 0.0 {
            return Err(format!("Sound speed must be positive, got {}", sound_speed));
        }
        if absorption_coefficient < 0.0 {
            return Err(format!(
                "Absorption coefficient must be non-negative, got {}",
                absorption_coefficient
            ));
        }
        if !(0.5..=3.0).contains(&absorption_power) {
            return Err(format!(
                "Absorption power must be in range [0.5, 3.0], got {}",
                absorption_power
            ));
        }
        if nonlinearity <= 0.0 {
            return Err(format!(
                "Nonlinearity parameter must be positive, got {}",
                nonlinearity
            ));
        }

        Ok(Self {
            density,
            sound_speed,
            absorption_coefficient,
            absorption_power,
            nonlinearity,
        })
    }

    /// Acoustic impedance Z = ρc (kg/m²s or Rayl)
    ///
    /// Determines reflection/transmission coefficients at interfaces.
    #[inline]
    #[must_use]
    pub fn impedance(&self) -> f64 {
        self.density * self.sound_speed
    }

    /// Absorption coefficient at frequency f (MHz) → α(f) = α₀ f^y (Np/m)
    ///
    /// # Arguments
    ///
    /// - `freq_mhz`: Frequency in MHz
    ///
    /// # Returns
    ///
    /// Absorption coefficient in Np/m (Nepers per meter)
    #[inline]
    #[must_use]
    pub fn absorption_at_frequency(&self, freq_mhz: f64) -> f64 {
        self.absorption_coefficient * freq_mhz.powf(self.absorption_power)
    }

    /// Nonlinearity coefficient β = 1 + B/(2A)
    ///
    /// Alternative parameterization used in some nonlinear wave equations.
    #[inline]
    #[must_use]
    pub fn nonlinearity_coefficient(&self) -> f64 {
        1.0 + self.nonlinearity / 2.0
    }

    /// Water properties at 20°C
    #[must_use]
    pub fn water() -> Self {
        Self {
            density: DENSITY_WATER,
            sound_speed: 1481.0,
            absorption_coefficient: 0.002,
            absorption_power: 2.0,
            nonlinearity: B_OVER_A_WATER, // 5.2 at 20°C (Duck 1990 Table 4.16)
        }
    }

    /// Soft tissue properties (generic)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            density: DENSITY_TISSUE,
            sound_speed: SOUND_SPEED_TISSUE,
            absorption_coefficient: 0.5,
            absorption_power: 1.1,
            nonlinearity: 6.5,
        }
    }

    /// Liver tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1060 kg/m³
    /// - Sound speed: ~1570 m/s
    /// - Attenuation: ~0.5 dB/(MHz·cm) = ~0.58 Np/(MHz·m)
    /// - B/A: 6.75 (Duck 1990 Table 4.16 mean)
    #[must_use]
    pub fn liver() -> Self {
        Self {
            density: DENSITY_LIVER,
            sound_speed: SOUND_SPEED_LIVER,
            absorption_coefficient: 0.58,
            absorption_power: 1.1,
            nonlinearity: B_OVER_A_LIVER,
        }
    }

    /// Brain tissue acoustic properties
    ///
    /// Based on clinical measurements (Duck 1990 Table 4.6/4.16):
    /// - Density: ~1040 kg/m³
    /// - Sound speed: ~1546 m/s
    /// - Attenuation: ~0.6 dB/(MHz·cm) = ~0.69 Np/(MHz·m)
    /// - B/A: 6.55 (Duck 1990 Table 4.16)
    #[must_use]
    pub fn brain() -> Self {
        Self {
            density: DENSITY_BRAIN,
            sound_speed: SOUND_SPEED_BRAIN,
            absorption_coefficient: 0.69,
            absorption_power: 1.0,
            nonlinearity: B_OVER_A_BRAIN,
        }
    }

    /// Kidney tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1050 kg/m³
    /// - Sound speed: ~1560 m/s
    /// - Attenuation: ~0.7 dB/(MHz·cm) = ~0.81 Np/(MHz·m)
    /// - B/A: 7.2 (Duck 1990 Table 4.16; SSOT B_OVER_A_KIDNEY)
    #[must_use]
    pub fn kidney() -> Self {
        Self {
            density: DENSITY_TISSUE,
            sound_speed: SOUND_SPEED_KIDNEY,
            absorption_coefficient: 0.81,
            absorption_power: 1.1,
            nonlinearity: B_OVER_A_KIDNEY,
        }
    }

    /// Muscle tissue acoustic properties
    ///
    /// Based on clinical measurements (Duck 1990 Table 4.1 and 4.6):
    /// - Density: ~1090 kg/m³ (IT'IS Foundation; upper Duck range 1041–1090)
    /// - Sound speed: ~1580 m/s (Duck 1990 Table 4.6 mean)
    /// - Attenuation: ~1.0 dB/(MHz·cm) = ~1.15 Np/(MHz·m)
    /// - B/A: 7.4 (Duck 1990 Table 4.16; SSOT B_OVER_A_MUSCLE)
    #[must_use]
    pub fn muscle() -> Self {
        Self {
            density: DENSITY_MUSCLE,
            sound_speed: SOUND_SPEED_MUSCLE,
            absorption_coefficient: 1.15,
            absorption_power: 1.0,
            nonlinearity: B_OVER_A_MUSCLE,
        }
    }

    /// Fat tissue acoustic properties
    ///
    /// Based on clinical measurements (Duck 1990 Table 4.1/4.6):
    /// - Density: ~928 kg/m³ (Duck 1990 mean; range 900–950 kg/m³)
    /// - Sound speed: ~1450 m/s (lower than soft tissue)
    /// - Attenuation: ~0.6 dB/(MHz·cm) = ~0.69 Np/(MHz·m)
    /// - B/A: 9.6 (Duck 1990 Table 4.16; SSOT B_OVER_A_FAT)
    #[must_use]
    pub fn fat() -> Self {
        Self {
            density: DENSITY_FAT,
            sound_speed: SOUND_SPEED_FAT,
            absorption_coefficient: 0.69,
            absorption_power: 1.0,
            nonlinearity: B_OVER_A_FAT,
        }
    }
}

impl fmt::Display for AcousticPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Acoustic(ρ={:.0} kg/m³, c={:.0} m/s, Z={:.2e} Rayl, α₀={:.3}, y={:.2}, B/A={:.1})",
            self.density,
            self.sound_speed,
            self.impedance(),
            self.absorption_coefficient,
            self.absorption_power,
            self.nonlinearity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

    #[test]
    fn test_acoustic_impedance() {
        let water = AcousticPropertyData::water();
        let expected_impedance = DENSITY_WATER * 1481.0; // ρc
        assert!((water.impedance() - expected_impedance).abs() < 1.0);
    }

    #[test]
    fn test_acoustic_absorption() {
        let props = AcousticPropertyData {
            density: DENSITY_WATER_NOMINAL,
            sound_speed: SOUND_SPEED_WATER_SIM,
            absorption_coefficient: 0.5,
            absorption_power: 1.1,
            nonlinearity: 5.0,
        };

        let alpha_1mhz = props.absorption_at_frequency(1.0);
        let alpha_2mhz = props.absorption_at_frequency(2.0);

        // α(2f) = α₀(2f)^y = 2^y α(f)
        let expected_ratio = 2.0_f64.powf(1.1);
        assert!((alpha_2mhz / alpha_1mhz - expected_ratio).abs() < 1e-10);
    }

    #[test]
    fn test_acoustic_validation() {
        // Negative density should fail
        assert!(AcousticPropertyData::new(-DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.1, 5.0).is_err());

        // Invalid absorption power should fail
        assert!(AcousticPropertyData::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 5.0, 5.0).is_err());

        // Valid parameters should succeed
        let props = AcousticPropertyData::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.1, 5.0).unwrap();
        assert!(props.density > 0.0);
    }
}
