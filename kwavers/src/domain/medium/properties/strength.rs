//! Mechanical strength property data structures
//!
//! # Mathematical Foundation
//!
//! ## Yield Criterion (Von Mises)
//!
//! Plastic deformation occurs when equivalent stress reaches yield strength:
//! ```text
//! σ_eq = √(3J₂) ≤ σ_y
//! ```
//! where J₂ is the second invariant of the deviatoric stress tensor.
//!
//! ## Fatigue Life (Basquin's Law)
//!
//! Number of cycles to failure:
//! ```text
//! N = C (Δσ)^(-b)
//! ```
//! - `N`: Cycles to failure
//! - `Δσ`: Stress amplitude
//! - `b`: Fatigue strength exponent
//! - `C`: Material constant
//!
//! ## Invariants
//!
//! - `yield_strength > 0`
//! - `ultimate_strength ≥ yield_strength`
//! - `hardness > 0`
//! - `fatigue_exponent > 0` (typical range: 5-15)

use std::fmt;

/// Canonical mechanical strength properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrengthPropertyData {
    /// Yield strength σ_y (Pa)
    ///
    /// Stress at which plastic deformation begins.
    pub yield_strength: f64,

    /// Ultimate tensile strength σ_u (Pa)
    ///
    /// Maximum stress before fracture.
    pub ultimate_strength: f64,

    /// Hardness H (Pa)
    ///
    /// Resistance to localized plastic deformation.
    /// Approximation: H ≈ 3σ_y for metals
    pub hardness: f64,

    /// Fatigue strength exponent b (dimensionless)
    ///
    /// Material constant in Basquin's fatigue law.
    /// Typical values:
    /// - Metals: 8-12
    /// - Ceramics: 10-20
    /// - Polymers: 5-10
    pub fatigue_exponent: f64,
}

impl StrengthPropertyData {
    /// Construct with validation
    pub fn new(
        yield_strength: f64,
        ultimate_strength: f64,
        hardness: f64,
        fatigue_exponent: f64,
    ) -> Result<Self, String> {
        if yield_strength <= 0.0 {
            return Err(format!(
                "Yield strength must be positive, got {}",
                yield_strength
            ));
        }
        if ultimate_strength < yield_strength {
            return Err(format!(
                "Ultimate strength ({}) must be ≥ yield strength ({})",
                ultimate_strength, yield_strength
            ));
        }
        if hardness <= 0.0 {
            return Err(format!("Hardness must be positive, got {}", hardness));
        }
        if fatigue_exponent <= 0.0 {
            return Err(format!(
                "Fatigue exponent must be positive, got {}",
                fatigue_exponent
            ));
        }

        Ok(Self {
            yield_strength,
            ultimate_strength,
            hardness,
            fatigue_exponent,
        })
    }

    /// Estimate hardness from yield strength (H ≈ 3σ_y for metals)
    pub fn estimate_hardness(yield_strength: f64) -> f64 {
        3.0 * yield_strength
    }

    /// Steel properties (mild steel)
    pub fn steel() -> Self {
        Self {
            yield_strength: 250e6,
            ultimate_strength: 400e6,
            hardness: 750e6,
            fatigue_exponent: 10.0,
        }
    }

    /// Bone properties (cortical bone)
    pub fn bone() -> Self {
        Self {
            yield_strength: 130e6,
            ultimate_strength: 150e6,
            hardness: 390e6,
            fatigue_exponent: 12.0,
        }
    }
}

impl fmt::Display for StrengthPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Strength(σ_y={:.0} MPa, σ_u={:.0} MPa, H={:.0} MPa, b={:.1})",
            self.yield_strength / 1e6,
            self.ultimate_strength / 1e6,
            self.hardness / 1e6,
            self.fatigue_exponent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strength_hardness_estimate() {
        let hardness = StrengthPropertyData::estimate_hardness(250e6);
        assert_eq!(hardness, 750e6);
    }

    #[test]
    fn test_strength_validation() {
        // Ultimate < yield should fail
        assert!(StrengthPropertyData::new(400e6, 250e6, 750e6, 10.0).is_err());

        // Valid parameters should succeed
        assert!(StrengthPropertyData::new(250e6, 400e6, 750e6, 10.0).is_ok());
    }
}
