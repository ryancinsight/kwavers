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

impl ElasticPropertyData {
    /// Construct from Lamé parameters with validation
    ///
    /// # Errors
    ///
    /// Returns error if parameters violate physical constraints
    pub fn new(density: f64, lambda: f64, mu: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if lambda < 0.0 {
            return Err(format!("Lamé lambda must be non-negative, got {}", lambda));
        }
        if mu <= 0.0 {
            return Err(format!("Shear modulus mu must be positive, got {}", mu));
        }

        // Check Poisson's ratio bounds: -1 < ν < 0.5
        let nu = lambda / (2.0 * (lambda + mu));
        if nu <= -1.0 || nu >= 0.5 {
            return Err(format!("Poisson's ratio {} violates bounds (-1, 0.5)", nu));
        }

        Ok(Self {
            density,
            lambda,
            mu,
        })
    }

    /// Construct from engineering parameters (Young's modulus E, Poisson's ratio ν)
    ///
    /// # Arguments
    ///
    /// - `density`: ρ (kg/m³)
    /// - `youngs_modulus`: E (Pa)
    /// - `poisson_ratio`: ν (dimensionless, must be in (-1, 0.5))
    ///
    /// # Panics
    ///
    /// Panics if parameters are unphysical (use `try_from_engineering` for fallible version)
    pub fn from_engineering(density: f64, youngs_modulus: f64, poisson_ratio: f64) -> Self {
        Self::try_from_engineering(density, youngs_modulus, poisson_ratio)
            .expect("Invalid engineering parameters")
    }

    /// Fallible version of `from_engineering`
    pub fn try_from_engineering(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if youngs_modulus <= 0.0 {
            return Err(format!(
                "Young's modulus must be positive, got {}",
                youngs_modulus
            ));
        }
        if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 {
            return Err(format!(
                "Poisson's ratio must be in (-1, 0.5), got {}",
                poisson_ratio
            ));
        }

        let lambda =
            youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        Ok(Self {
            density,
            lambda,
            mu,
        })
    }

    /// Young's modulus E = μ(3λ + 2μ)/(λ + μ) (Pa)
    #[inline]
    pub fn youngs_modulus(&self) -> f64 {
        self.mu * (3.0 * self.lambda + 2.0 * self.mu) / (self.lambda + self.mu)
    }

    /// Poisson's ratio ν = λ/(2(λ + μ)) (dimensionless)
    #[inline]
    pub fn poisson_ratio(&self) -> f64 {
        self.lambda / (2.0 * (self.lambda + self.mu))
    }

    /// Bulk modulus K = λ + 2μ/3 (Pa)
    #[inline]
    pub fn bulk_modulus(&self) -> f64 {
        self.lambda + 2.0 * self.mu / 3.0
    }

    /// Shear modulus (alias for μ)
    #[inline]
    pub fn shear_modulus(&self) -> f64 {
        self.mu
    }

    /// P-wave (compressional) speed c_p = √((λ + 2μ)/ρ) (m/s)
    #[inline]
    pub fn p_wave_speed(&self) -> f64 {
        ((self.lambda + 2.0 * self.mu) / self.density).sqrt()
    }

    /// S-wave (shear) speed c_s = √(μ/ρ) (m/s)
    #[inline]
    pub fn s_wave_speed(&self) -> f64 {
        (self.mu / self.density).sqrt()
    }

    /// Steel properties (generic)
    pub fn steel() -> Self {
        Self::from_engineering(7850.0, 200e9, 0.3)
    }

    /// Aluminum properties (generic)
    pub fn aluminum() -> Self {
        Self::from_engineering(2700.0, 69e9, 0.33)
    }

    /// Bone properties (cortical bone)
    pub fn bone() -> Self {
        Self::from_engineering(1850.0, 17e9, 0.3)
    }

    /// Construct from wave speeds (inverse problem)
    ///
    /// # Arguments
    ///
    /// - `density`: ρ (kg/m³)
    /// - `p_speed`: P-wave speed c_p (m/s)
    /// - `s_speed`: S-wave speed c_s (m/s)
    ///
    /// # Panics
    ///
    /// Panics if parameters are unphysical (use `try_from_wave_speeds` for fallible version)
    pub fn from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Self {
        Self::try_from_wave_speeds(density, p_speed, s_speed)
            .expect("Invalid wave speed parameters")
    }

    /// Fallible version of `from_wave_speeds`
    ///
    /// Recovers Lamé parameters from measured wave speeds:
    /// ```text
    /// μ = ρ c_s²
    /// λ = ρ c_p² - 2μ
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `density ≤ 0`
    /// - `p_speed ≤ 0` or `s_speed ≤ 0`
    /// - `s_speed ≥ p_speed` (physical constraint: shear waves are slower)
    pub fn try_from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if p_speed <= 0.0 {
            return Err(format!("P-wave speed must be positive, got {}", p_speed));
        }
        if s_speed <= 0.0 {
            return Err(format!("S-wave speed must be positive, got {}", s_speed));
        }
        if s_speed >= p_speed {
            return Err(format!(
                "S-wave speed ({}) must be less than P-wave speed ({})",
                s_speed, p_speed
            ));
        }

        // Recover Lamé parameters from wave speeds
        let mu = density * s_speed * s_speed;
        let lambda = density * p_speed * p_speed - 2.0 * mu;

        Self::new(density, lambda, mu)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_engineering_conversion() {
        let density = 7850.0; // kg/m³
        let youngs = 200e9; // Pa
        let poisson = 0.3;

        let elastic = ElasticPropertyData::from_engineering(density, youngs, poisson);

        // Verify round-trip conversion
        assert!((elastic.youngs_modulus() - youngs).abs() / youngs < 1e-10);
        assert!((elastic.poisson_ratio() - poisson).abs() < 1e-10);
    }

    #[test]
    fn test_elastic_wave_speeds() {
        let steel = ElasticPropertyData::steel();

        // Steel P-wave speed should be ~5960 m/s (typical range: 5800-6100 m/s)
        let cp = steel.p_wave_speed();
        assert!(
            cp > 5000.0 && cp < 7000.0,
            "P-wave speed {} out of expected range",
            cp
        );

        // S-wave speed should be less than P-wave speed
        let cs = steel.s_wave_speed();
        assert!(cs < cp);

        // S-wave speed should be ~3200 m/s (typical range: 3000-3400 m/s)
        assert!(
            cs > 2500.0 && cs < 4000.0,
            "S-wave speed {} out of expected range",
            cs
        );
    }

    #[test]
    fn test_elastic_poisson_bounds() {
        let density = 1000.0;

        // ν = 0.5 (incompressible) should fail
        assert!(ElasticPropertyData::try_from_engineering(density, 1e9, 0.5).is_err());

        // ν = -1 should fail
        assert!(ElasticPropertyData::try_from_engineering(density, 1e9, -1.0).is_err());

        // Valid ν = 0.3 should succeed
        assert!(ElasticPropertyData::try_from_engineering(density, 1e9, 0.3).is_ok());
    }

    #[test]
    fn test_elastic_moduli_relations() {
        let elastic = ElasticPropertyData::from_engineering(7850.0, 200e9, 0.3);

        let e = elastic.youngs_modulus();
        let nu = elastic.poisson_ratio();
        let k = elastic.bulk_modulus();
        let mu = elastic.shear_modulus();

        // K = E / (3(1 - 2ν))
        let k_expected = e / (3.0 * (1.0 - 2.0 * nu));
        assert!((k - k_expected).abs() / k < 1e-10);

        // μ = E / (2(1 + ν))
        let mu_expected = e / (2.0 * (1.0 + nu));
        assert!((mu - mu_expected).abs() / mu < 1e-10);
    }

    #[test]
    fn test_elastic_from_wave_speeds() {
        let density = 7850.0; // Steel density
        let cp = 5960.0; // P-wave speed
        let cs = 3220.0; // S-wave speed

        let elastic = ElasticPropertyData::from_wave_speeds(density, cp, cs);

        // Verify wave speeds are recovered
        assert!((elastic.p_wave_speed() - cp).abs() < 1e-6);
        assert!((elastic.s_wave_speed() - cs).abs() < 1e-6);

        // Verify density
        assert_eq!(elastic.density, density);

        // Verify Lamé parameters are positive
        assert!(elastic.lambda > 0.0);
        assert!(elastic.mu > 0.0);
    }

    #[test]
    fn test_elastic_from_wave_speeds_validation() {
        let density = 1000.0;

        // S-wave speed >= P-wave speed should fail (physical constraint)
        assert!(ElasticPropertyData::try_from_wave_speeds(density, 1500.0, 1600.0).is_err());

        // Negative density should fail
        assert!(ElasticPropertyData::try_from_wave_speeds(-1000.0, 1500.0, 1000.0).is_err());

        // Negative wave speeds should fail
        assert!(ElasticPropertyData::try_from_wave_speeds(density, -1500.0, 1000.0).is_err());
        assert!(ElasticPropertyData::try_from_wave_speeds(density, 1500.0, -1000.0).is_err());

        // Valid parameters should succeed
        assert!(ElasticPropertyData::try_from_wave_speeds(density, 1500.0, 1000.0).is_ok());
    }

    #[test]
    fn test_elastic_wave_speed_round_trip() {
        // Start with engineering parameters
        let original = ElasticPropertyData::from_engineering(2700.0, 69e9, 0.33);

        // Extract wave speeds
        let cp = original.p_wave_speed();
        let cs = original.s_wave_speed();

        // Reconstruct from wave speeds
        let reconstructed = ElasticPropertyData::from_wave_speeds(original.density, cp, cs);

        // Verify Lamé parameters match (within numerical tolerance)
        assert!((reconstructed.lambda - original.lambda).abs() / original.lambda < 1e-10);
        assert!((reconstructed.mu - original.mu).abs() / original.mu < 1e-10);

        // Verify derived properties match
        assert!(
            (reconstructed.youngs_modulus() - original.youngs_modulus()).abs()
                / original.youngs_modulus()
                < 1e-10
        );
        assert!((reconstructed.poisson_ratio() - original.poisson_ratio()).abs() < 1e-10);
    }
}
