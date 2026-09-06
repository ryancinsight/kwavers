//! Derived elastic identities — all delegated to `proteus::elastic::IsotropicModuli`.
//!
//! This module owns no algebra. Every formula here is the provider's; kwavers
//! only adapts the stored Lamé pair and density into the typed call and unwraps
//! the quantity back to `f64` for the existing medium API.

use aequitas::systems::si::quantities::{MassDensity, Pressure};
use proteus::elastic::IsotropicModuli;

use super::ElasticPropertyData;

impl ElasticPropertyData {
    /// Rebuild the provider moduli from the stored Lamé pair.
    ///
    /// Every construction path (`new`, `try_from_engineering`,
    /// `try_from_wave_speeds`) already validated the pair against the
    /// provider's positive-definite domain. A panic here means the public
    /// fields were mutated into a state those constructors would have
    /// rejected — a caller bug, not a derivation failure.
    #[inline]
    fn isotropic_moduli(&self) -> IsotropicModuli<f64> {
        IsotropicModuli::<f64>::from_lame(
            Pressure::from_base(self.lambda),
            Pressure::from_base(self.mu),
        )
        .expect(
            "ElasticPropertyData Lamé pair is validated at construction; \
             mutating public fields outside the positive-definite domain is a caller bug",
        )
    }

    /// Young's modulus `E = μ(3λ + 2μ)/(λ + μ)` (Pa).
    #[inline]
    #[must_use]
    pub fn youngs_modulus(&self) -> f64 {
        *self.isotropic_moduli().youngs_modulus().as_base()
    }

    /// Poisson's ratio `ν = λ/(2(λ + μ))` (dimensionless).
    #[inline]
    #[must_use]
    pub fn poisson_ratio(&self) -> f64 {
        *self.isotropic_moduli().poissons_ratio().as_base()
    }

    /// Bulk modulus `K = λ + 2μ/3` (Pa).
    #[inline]
    #[must_use]
    pub fn bulk_modulus(&self) -> f64 {
        *self.isotropic_moduli().bulk_modulus().as_base()
    }

    /// Shear modulus `μ` (Pa).
    #[inline]
    #[must_use]
    pub fn shear_modulus(&self) -> f64 {
        *self.isotropic_moduli().shear_modulus().as_base()
    }

    /// P-wave (compressional) speed `c_p = √((λ + 2μ)/ρ)` (m/s).
    ///
    /// Density was validated positive at construction; a panic here has the
    /// same meaning as [`Self::isotropic_moduli`].
    #[inline]
    #[must_use]
    pub fn p_wave_speed(&self) -> f64 {
        *self
            .isotropic_moduli()
            .compressional_wave_speed(MassDensity::from_base(self.density))
            .expect(
                "ElasticPropertyData density is validated positive at construction; \
                 mutating it to a non-positive value is a caller bug",
            )
            .as_base()
    }

    /// S-wave (shear) speed `c_s = √(μ/ρ)` (m/s).
    #[inline]
    #[must_use]
    pub fn s_wave_speed(&self) -> f64 {
        *self
            .isotropic_moduli()
            .shear_wave_speed(MassDensity::from_base(self.density))
            .expect(
                "ElasticPropertyData density is validated positive at construction; \
                 mutating it to a non-positive value is a caller bug",
            )
            .as_base()
    }
}
