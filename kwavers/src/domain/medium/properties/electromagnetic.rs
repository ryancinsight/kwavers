//! Electromagnetic material property data structures
//!
//! # Mathematical Foundation
//!
//! Maxwell's equations in matter:
//! ```text
//! ∇ × E = -∂B/∂t
//! ∇ × H = J + ∂D/∂t
//! ∇ · D = ρ_charge
//! ∇ · B = 0
//! ```
//!
//! Constitutive relations:
//! ```text
//! D = ε₀ε_r E  (electric displacement)
//! B = μ₀μ_r H  (magnetic flux density)
//! J = σE       (Ohm's law)
//! ```
//!
//! ## Wave Speed
//!
//! Electromagnetic wave speed: c = c₀/√(ε_r μ_r)
//!
//! ## Impedance
//!
//! Intrinsic impedance: Z = Z₀√(μ_r/ε_r)
//!
//! ## Invariants
//!
//! - `permittivity ≥ 1.0` (vacuum is lower bound)
//! - `permeability ≥ 1.0` (most materials)
//! - `conductivity ≥ 0.0`

use std::fmt;

/// Canonical electromagnetic material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectromagneticPropertyData {
    /// Relative permittivity ε_r (dimensionless)
    pub permittivity: f64,

    /// Relative permeability μ_r (dimensionless)
    pub permeability: f64,

    /// Electrical conductivity σ (S/m)
    pub conductivity: f64,

    /// Dielectric relaxation time τ (s)
    pub relaxation_time: Option<f64>,
}

impl ElectromagneticPropertyData {
    /// Construct with validation
    pub fn new(
        permittivity: f64,
        permeability: f64,
        conductivity: f64,
        relaxation_time: Option<f64>,
    ) -> Result<Self, String> {
        if permittivity < 1.0 {
            return Err(format!(
                "Relative permittivity must be ≥ 1.0, got {}",
                permittivity
            ));
        }
        if permeability < 0.0 {
            return Err(format!(
                "Relative permeability must be non-negative, got {}",
                permeability
            ));
        }
        if conductivity < 0.0 {
            return Err(format!(
                "Conductivity must be non-negative, got {}",
                conductivity
            ));
        }
        if let Some(tau) = relaxation_time {
            if tau <= 0.0 {
                return Err(format!("Relaxation time must be positive, got {}", tau));
            }
        }

        Ok(Self {
            permittivity,
            permeability,
            conductivity,
            relaxation_time,
        })
    }

    /// Electromagnetic wave speed c = c₀/√(ε_r μ_r) (m/s)
    #[inline]
    pub fn wave_speed(&self) -> f64 {
        const C0: f64 = 299_792_458.0;
        C0 / (self.permittivity * self.permeability).sqrt()
    }

    /// Intrinsic impedance Z = Z₀√(μ_r/ε_r) (Ω)
    #[inline]
    pub fn impedance(&self) -> f64 {
        const Z0: f64 = 376.730_313_668;
        Z0 * (self.permeability / self.permittivity).sqrt()
    }

    /// Refractive index n = √(ε_r μ_r)
    #[inline]
    pub fn refractive_index(&self) -> f64 {
        (self.permittivity * self.permeability).sqrt()
    }

    /// Skin depth δ = √(2/(ωμσ)) at angular frequency ω
    pub fn skin_depth(&self, frequency_hz: f64) -> f64 {
        if self.conductivity == 0.0 {
            return f64::INFINITY;
        }
        const MU0: f64 = 1.25663706212e-6;
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let mu = MU0 * self.permeability;
        (2.0 / (omega * mu * self.conductivity)).sqrt()
    }

    /// Vacuum properties
    pub fn vacuum() -> Self {
        Self {
            permittivity: 1.0,
            permeability: 1.0,
            conductivity: 0.0,
            relaxation_time: None,
        }
    }

    /// Water properties (at RF frequencies)
    pub fn water() -> Self {
        Self {
            permittivity: 80.0,
            permeability: 1.0,
            conductivity: 0.005,
            relaxation_time: Some(8.3e-12),
        }
    }

    /// Biological tissue properties (generic)
    pub fn tissue() -> Self {
        Self {
            permittivity: 50.0,
            permeability: 1.0,
            conductivity: 0.5,
            relaxation_time: Some(10e-12),
        }
    }
}

impl fmt::Display for ElectromagneticPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EM(ε_r={:.1}, μ_r={:.1}, σ={:.3} S/m, c={:.2e} m/s, n={:.2})",
            self.permittivity,
            self.permeability,
            self.conductivity,
            self.wave_speed(),
            self.refractive_index()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_wave_speed() {
        let water = ElectromagneticPropertyData::water();
        const C0: f64 = 299_792_458.0;
        let expected = C0 / (water.permittivity * water.permeability).sqrt();
        assert!((water.wave_speed() - expected).abs() < 1.0);
    }

    #[test]
    fn test_em_refractive_index() {
        let water = ElectromagneticPropertyData::water();
        assert!((water.refractive_index() - 80.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_em_skin_depth() {
        let tissue = ElectromagneticPropertyData::tissue();
        let delta = tissue.skin_depth(1e9);
        assert!(delta > 0.0 && delta < 1.0);
    }

    #[test]
    fn test_em_validation() {
        assert!(ElectromagneticPropertyData::new(0.5, 1.0, 0.0, None).is_err());
        assert!(ElectromagneticPropertyData::new(1.0, 1.0, -1.0, None).is_err());
        assert!(ElectromagneticPropertyData::new(1.0, 1.0, 0.0, None).is_ok());
    }
}
