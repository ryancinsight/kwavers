//! Elastic material property data structures
//!
//! # Mathematical Foundation
//!
//! Stress-strain relation (Hooke's law for isotropic linear elasticity):
//! ```text
//! σ = λ tr(ε)I + 2με
//! ```
//!
//! Where:
//! - `σ`: Stress tensor (Pa)
//! - `ε`: Strain tensor (dimensionless)
//! - `λ`: Lamé's first parameter (Pa)
//! - `μ`: Lamé's second parameter (shear modulus) (Pa)
//! - `I`: Identity tensor
//!
//! ## Wave Speeds
//!
//! P-wave (compressional): `c_p = √((λ + 2μ)/ρ)`
//! S-wave (shear): `c_s = √(μ/ρ)`
//!
//! ## Engineering Parameters
//!
//! Relationships to Young's modulus E and Poisson's ratio ν:
//! ```text
//! λ = Eν / ((1+ν)(1-2ν))
//! μ = E / (2(1+ν))
//! K = λ + 2μ/3  (bulk modulus)
//! ```
//!
//! ## Invariants
//!
//! - `density > 0`
//! - `lambda ≥ 0`
//! - `mu > 0`
//! - `-1 < ν < 0.5` (Poisson's ratio bounds)
//! - `E > 0` (Young's modulus)

mod computed;
mod constructors;
#[cfg(test)]
mod tests;

use std::fmt;

/// Canonical elastic material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticPropertyData {
    /// Density ρ (kg/m³)
    pub density: f64,

    /// Lamé first parameter λ (Pa)
    ///
    /// Related to bulk compressibility. Can be zero for some materials.
    pub lambda: f64,

    /// Lamé second parameter μ (shear modulus) (Pa)
    ///
    /// Resistance to shear deformation. Must be positive.
    pub mu: f64,
}

impl fmt::Display for ElasticPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Elastic(ρ={:.0} kg/m³, E={:.2e} Pa, ν={:.3}, c_p={:.0} m/s, c_s={:.0} m/s)",
            self.density,
            self.youngs_modulus(),
            self.poisson_ratio(),
            self.p_wave_speed(),
            self.s_wave_speed()
        )
    }
}
