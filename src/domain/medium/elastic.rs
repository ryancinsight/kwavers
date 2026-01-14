//! Elastic properties trait for solid media
//!
//! This module defines traits for elastic wave propagation in solid media,
//! including Lamé parameters and wave speeds.

use crate::domain::grid::Grid;
use crate::domain::medium::core::{ArrayAccess, CoreMedium};
use ndarray::Array3;

/// Trait for elastic medium properties
pub trait ElasticProperties: CoreMedium {
    /// Returns Lamé's first parameter λ (Pa)
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Returns Lamé's second parameter μ (shear modulus) (Pa)
    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Calculates shear wave speed (m/s)
    fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.lame_mu(x, y, z, grid);
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let rho = self.density(i, j, k);
        if rho > 0.0 {
            (mu / rho).sqrt()
        } else {
            0.0
        }
    }

    /// Calculates compressional wave speed (m/s)
    fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let lambda = self.lame_lambda(x, y, z, grid);
        let mu = self.lame_mu(x, y, z, grid);
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let rho = self.density(i, j, k);
        if rho > 0.0 {
            ((lambda + 2.0 * mu) / rho).sqrt()
        } else {
            0.0
        }
    }
}

/// Trait for array-based elastic property access
pub trait ElasticArrayAccess: ElasticProperties + ArrayAccess {
    /// Returns a 3D array of Lamé's first parameter λ values (Pa)
    fn lame_lambda_array(&self) -> Array3<f64>;

    /// Returns a 3D array of Lamé's second parameter μ values (Pa)
    fn lame_mu_array(&self) -> Array3<f64>;

    /// Returns a 3D array of shear wave speeds (m/s)
    ///
    /// # Mathematical Specification
    ///
    /// Shear wave speed in elastic medium:
    /// ```text
    /// c_s = sqrt(μ / ρ)
    /// ```
    /// where:
    /// - `μ` is Lamé's second parameter (shear modulus, Pa)
    /// - `ρ` is mass density (kg/m³)
    ///
    /// # Physical Validity
    ///
    /// For biological tissues, shear wave speed typically ranges from 0.5 to 10 m/s.
    /// For hard tissues (bone, cartilage), it may reach 1000-2000 m/s.
    ///
    /// # Implementation Requirements
    ///
    /// This method must be implemented by all concrete types. There is no default
    /// implementation to prevent silent failures from zero-valued shear speeds.
    ///
    /// Implementations should:
    /// 1. Compute `c_s = sqrt(μ / ρ)` for each grid point
    /// 2. Handle zero density: return 0.0 or appropriate fallback
    /// 3. Validate: ensure c_s ≥ 0 for all elements
    ///
    /// # References
    ///
    /// - Landau & Lifshitz, "Theory of Elasticity" (1986), §24
    /// - Graff, "Wave Motion in Elastic Solids" (1975), Ch. 1
    fn shear_sound_speed_array(&self) -> Array3<f64>;

    /// Returns a 3D array of shear viscosity coefficients (Pa·s)
    ///
    /// # Default Behavior: Lossless Elastic Medium
    ///
    /// Returns zero by default, representing a perfectly elastic (lossless) medium
    /// with no viscous attenuation. This is physically valid for:
    /// - Idealized elastic wave propagation studies
    /// - High-frequency waves where viscous effects are negligible
    /// - Comparison with analytical solutions in lossless media
    ///
    /// # Mathematical Specification
    ///
    /// Viscoelastic shear stress tensor (Kelvin-Voigt model):
    /// ```text
    /// τ_ij = μ ∂u_i/∂x_j + η_s ∂(∂u_i/∂x_j)/∂t
    /// ```
    /// where:
    /// - `η_s` is shear viscosity coefficient (Pa·s)
    /// - For biological tissues: `η_s ≈ 0.001-1.0 Pa·s`
    /// - For bone/cartilage: `η_s ≈ 10-100 Pa·s`
    ///
    /// # Viscoelastic Media
    ///
    /// **Warning**: For realistic tissue modeling with attenuation, implementations
    /// must override this method with non-zero viscosity values. Zero viscosity
    /// leads to:
    /// - No amplitude decay with propagation distance
    /// - Unrealistic wave behavior in biological tissues
    /// - Incorrect phase velocity at high frequencies
    ///
    /// # Computing from Q-Factor
    ///
    /// Shear viscosity can be computed from quality factor Q:
    /// ```text
    /// η_s = μ / (ω · Q)
    /// ```
    /// where:
    /// - `ω` is angular frequency (rad/s)
    /// - `Q` is quality factor (dimensionless, typically 10-100 for soft tissues)
    ///
    /// # References
    ///
    /// - Fung, "Biomechanics: Mechanical Properties of Living Tissues" (1993)
    /// - Catheline et al., "Measurement of viscoelastic properties of homogeneous
    ///   soft solid using transient elastography" (2004), Ultrasound Med. Biol. 30(11), 1461-1469
    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        let shape = self.lame_mu_array().dim();
        Array3::zeros(shape)
    }

    /// Returns a 3D array of bulk viscosity coefficients
    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        let shape = self.lame_mu_array().dim();
        Array3::zeros(shape)
    }
}
