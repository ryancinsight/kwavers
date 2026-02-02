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
    pub fn absorption_at_frequency(&self, freq_mhz: f64) -> f64 {
        self.absorption_coefficient * freq_mhz.powf(self.absorption_power)
    }

    /// Nonlinearity coefficient β = 1 + B/(2A)
    ///
    /// Alternative parameterization used in some nonlinear wave equations.
    #[inline]
    pub fn nonlinearity_coefficient(&self) -> f64 {
        1.0 + self.nonlinearity / 2.0
    }

    /// Water properties at 20°C
    pub fn water() -> Self {
        Self {
            density: 998.0,
            sound_speed: 1481.0,
            absorption_coefficient: 0.002,
            absorption_power: 2.0,
            nonlinearity: 5.0,
        }
    }

    /// Soft tissue properties (generic)
    pub fn soft_tissue() -> Self {
        Self {
            density: 1050.0,
            sound_speed: 1540.0,
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
    /// - B/A: ~6.8
    pub fn liver() -> Self {
        Self {
            density: 1060.0,
            sound_speed: 1570.0,
            absorption_coefficient: 0.58,
            absorption_power: 1.1,
            nonlinearity: 6.8,
        }
    }

    /// Brain tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1040 kg/m³
    /// - Sound speed: ~1540 m/s
    /// - Attenuation: ~0.6 dB/(MHz·cm) = ~0.69 Np/(MHz·m)
    /// - B/A: ~6.5
    pub fn brain() -> Self {
        Self {
            density: 1040.0,
            sound_speed: 1540.0,
            absorption_coefficient: 0.69,
            absorption_power: 1.0,
            nonlinearity: 6.5,
        }
    }

    /// Kidney tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1050 kg/m³
    /// - Sound speed: ~1560 m/s
    /// - Attenuation: ~0.7 dB/(MHz·cm) = ~0.81 Np/(MHz·m)
    /// - B/A: ~6.7
    pub fn kidney() -> Self {
        Self {
            density: 1050.0,
            sound_speed: 1560.0,
            absorption_coefficient: 0.81,
            absorption_power: 1.1,
            nonlinearity: 6.7,
        }
    }

    /// Muscle tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1070 kg/m³
    /// - Sound speed: ~1580 m/s
    /// - Attenuation: ~1.0 dB/(MHz·cm) = ~1.15 Np/(MHz·m)
    /// - B/A: ~7.4
    pub fn muscle() -> Self {
        Self {
            density: 1070.0,
            sound_speed: 1580.0,
            absorption_coefficient: 1.15,
            absorption_power: 1.0,
            nonlinearity: 7.4,
        }
    }

    /// Fat tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~950 kg/m³ (lower than water)
    /// - Sound speed: ~1450 m/s (lower than soft tissue)
    /// - Attenuation: ~0.6 dB/(MHz·cm) = ~0.69 Np/(MHz·m)
    /// - B/A: ~10.0 (higher nonlinearity than soft tissue)
    pub fn fat() -> Self {
        Self {
            density: 950.0,
            sound_speed: 1450.0,
            absorption_coefficient: 0.69,
            absorption_power: 1.0,
            nonlinearity: 10.0,
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

    #[test]
    fn test_acoustic_impedance() {
        let water = AcousticPropertyData::water();
        let expected_impedance = 998.0 * 1481.0; // ρc
        assert!((water.impedance() - expected_impedance).abs() < 1.0);
    }

    #[test]
    fn test_acoustic_absorption() {
        let props = AcousticPropertyData {
            density: 1000.0,
            sound_speed: 1500.0,
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
        assert!(AcousticPropertyData::new(-1000.0, 1500.0, 0.5, 1.1, 5.0).is_err());

        // Invalid absorption power should fail
        assert!(AcousticPropertyData::new(1000.0, 1500.0, 0.5, 5.0, 5.0).is_err());

        // Valid parameters should succeed
        assert!(AcousticPropertyData::new(1000.0, 1500.0, 0.5, 1.1, 5.0).is_ok());
    }
}
