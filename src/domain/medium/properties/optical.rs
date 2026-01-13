//! Optical material property data structures for light propagation and scattering
//!
//! # Mathematical Foundation
//!
//! ## Radiative Transfer Equation (RTE)
//!
//! Light propagation in scattering media:
//! ```text
//! dI/ds = -μ_t I + μ_s ∫ p(θ) I(s') dΩ'
//! ```
//!
//! Where:
//! - `I`: Radiance (W/m²/sr)
//! - `s`: Path length (m)
//! - `μ_t = μ_a + μ_s`: Total attenuation coefficient (m⁻¹)
//! - `μ_a`: Absorption coefficient (m⁻¹)
//! - `μ_s`: Scattering coefficient (m⁻¹)
//! - `p(θ)`: Phase function (angular scattering probability)
//!
//! ## Henyey-Greenstein Phase Function
//!
//! Anisotropic scattering model:
//! ```text
//! p(θ) = (1 - g²) / (4π (1 + g² - 2g cos θ)^(3/2))
//! ```
//! - `g`: Anisotropy factor (⟨cos θ⟩)
//! - `g = 0`: Isotropic scattering
//! - `g > 0`: Forward scattering (typical for biological tissue)
//! - `g < 0`: Backward scattering
//!
//! ## Invariants
//!
//! - `absorption_coefficient ≥ 0` (m⁻¹)
//! - `scattering_coefficient ≥ 0` (m⁻¹)
//! - `-1 ≤ anisotropy ≤ 1` (dimensionless)
//! - `refractive_index ≥ 1.0` (vacuum is lower bound)

use std::fmt;

/// Canonical optical material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalPropertyData {
    /// Absorption coefficient μ_a (m⁻¹)
    pub absorption_coefficient: f64,

    /// Scattering coefficient μ_s (m⁻¹)
    pub scattering_coefficient: f64,

    /// Anisotropy factor g = ⟨cos θ⟩ (dimensionless)
    ///
    /// Physical range for biological tissue: 0.7-0.99 (highly forward-scattering)
    pub anisotropy: f64,

    /// Refractive index n (dimensionless)
    ///
    /// Ratio of light speed in vacuum to light speed in medium: n = c₀/c
    pub refractive_index: f64,
}

impl OpticalPropertyData {
    /// Construct with validation of physical constraints
    pub fn new(
        absorption_coefficient: f64,
        scattering_coefficient: f64,
        anisotropy: f64,
        refractive_index: f64,
    ) -> Result<Self, String> {
        if absorption_coefficient < 0.0 {
            return Err(format!(
                "Absorption coefficient must be non-negative, got {}",
                absorption_coefficient
            ));
        }
        if scattering_coefficient < 0.0 {
            return Err(format!(
                "Scattering coefficient must be non-negative, got {}",
                scattering_coefficient
            ));
        }
        if !(-1.0..=1.0).contains(&anisotropy) {
            return Err(format!(
                "Anisotropy factor must be in range [-1, 1], got {}",
                anisotropy
            ));
        }
        if refractive_index < 1.0 {
            return Err(format!(
                "Refractive index must be ≥ 1.0 (vacuum limit), got {}",
                refractive_index
            ));
        }

        Ok(Self {
            absorption_coefficient,
            scattering_coefficient,
            anisotropy,
            refractive_index,
        })
    }

    /// Total attenuation coefficient μ_t = μ_a + μ_s (m⁻¹)
    #[inline]
    pub fn total_attenuation(&self) -> f64 {
        self.absorption_coefficient + self.scattering_coefficient
    }

    /// Reduced scattering coefficient μ_s' = μ_s (1 - g) (m⁻¹)
    #[inline]
    pub fn reduced_scattering(&self) -> f64 {
        self.scattering_coefficient * (1.0 - self.anisotropy)
    }

    /// Optical penetration depth δ = 1/μ_eff (m)
    #[inline]
    pub fn penetration_depth(&self) -> f64 {
        let mu_s_prime = self.reduced_scattering();
        let mu_eff =
            (3.0 * self.absorption_coefficient * (self.absorption_coefficient + mu_s_prime)).sqrt();
        if mu_eff > 0.0 {
            1.0 / mu_eff
        } else {
            f64::INFINITY
        }
    }

    /// Mean free path l_mfp = 1/μ_t (m)
    #[inline]
    pub fn mean_free_path(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 {
            1.0 / mu_t
        } else {
            f64::INFINITY
        }
    }

    /// Transport mean free path l_tr = 1/(μ_a + μ_s') (m)
    #[inline]
    pub fn transport_mean_free_path(&self) -> f64 {
        let mu_tr = self.absorption_coefficient + self.reduced_scattering();
        if mu_tr > 0.0 {
            1.0 / mu_tr
        } else {
            f64::INFINITY
        }
    }

    /// Albedo α = μ_s / μ_t (dimensionless)
    #[inline]
    pub fn albedo(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 {
            self.scattering_coefficient / mu_t
        } else {
            0.0
        }
    }

    /// Fresnel reflectance at normal incidence R₀
    #[inline]
    pub fn fresnel_reflectance_normal(&self) -> f64 {
        let n1 = 1.0;
        let n2 = self.refractive_index;
        let r = (n1 - n2) / (n1 + n2);
        r * r
    }

    /// Water optical properties (visible spectrum, ~550 nm)
    pub fn water() -> Self {
        Self {
            absorption_coefficient: 0.01,
            scattering_coefficient: 0.001,
            anisotropy: 0.0,
            refractive_index: 1.33,
        }
    }

    /// Soft tissue optical properties (generic, ~650 nm)
    pub fn soft_tissue() -> Self {
        Self {
            absorption_coefficient: 0.5,
            scattering_coefficient: 100.0,
            anisotropy: 0.9,
            refractive_index: 1.4,
        }
    }

    /// Blood optical properties (oxygenated, ~650 nm)
    pub fn blood_oxygenated() -> Self {
        Self {
            absorption_coefficient: 50.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.95,
            refractive_index: 1.4,
        }
    }

    /// Blood optical properties (deoxygenated, ~650 nm)
    pub fn blood_deoxygenated() -> Self {
        Self {
            absorption_coefficient: 80.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.95,
            refractive_index: 1.4,
        }
    }

    /// Tumor tissue optical properties (hypervascular, ~650 nm)
    pub fn tumor() -> Self {
        Self {
            absorption_coefficient: 10.0,
            scattering_coefficient: 120.0,
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }

    /// Brain tissue optical properties (gray matter, ~650 nm)
    pub fn brain_gray_matter() -> Self {
        Self {
            absorption_coefficient: 0.8,
            scattering_coefficient: 150.0,
            anisotropy: 0.9,
            refractive_index: 1.38,
        }
    }

    /// Brain tissue optical properties (white matter, ~650 nm)
    pub fn brain_white_matter() -> Self {
        Self {
            absorption_coefficient: 1.0,
            scattering_coefficient: 250.0,
            anisotropy: 0.92,
            refractive_index: 1.38,
        }
    }

    /// Liver tissue optical properties (~650 nm)
    pub fn liver() -> Self {
        Self {
            absorption_coefficient: 2.0,
            scattering_coefficient: 120.0,
            anisotropy: 0.88,
            refractive_index: 1.39,
        }
    }

    /// Muscle tissue optical properties (~650 nm)
    pub fn muscle() -> Self {
        Self {
            absorption_coefficient: 0.8,
            scattering_coefficient: 100.0,
            anisotropy: 0.85,
            refractive_index: 1.37,
        }
    }

    /// Skin (epidermis) optical properties (~650 nm)
    pub fn skin_epidermis() -> Self {
        Self {
            absorption_coefficient: 5.0,
            scattering_coefficient: 300.0,
            anisotropy: 0.8,
            refractive_index: 1.4,
        }
    }

    /// Skin (dermis) optical properties (~650 nm)
    pub fn skin_dermis() -> Self {
        Self {
            absorption_coefficient: 1.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }

    /// Bone (cortical) optical properties (~650 nm)
    pub fn bone_cortical() -> Self {
        Self {
            absorption_coefficient: 5.0,
            scattering_coefficient: 500.0,
            anisotropy: 0.9,
            refractive_index: 1.55,
        }
    }

    /// Fat tissue optical properties (~650 nm)
    pub fn fat() -> Self {
        Self {
            absorption_coefficient: 0.3,
            scattering_coefficient: 100.0,
            anisotropy: 0.9,
            refractive_index: 1.46,
        }
    }
}

impl fmt::Display for OpticalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Optical(μ_a={:.2} m⁻¹, μ_s={:.1} m⁻¹, μ_s'={:.1} m⁻¹, g={:.3}, n={:.3}, δ={:.1} mm, α={:.3})",
            self.absorption_coefficient,
            self.scattering_coefficient,
            self.reduced_scattering(),
            self.anisotropy,
            self.refractive_index,
            self.penetration_depth() * 1000.0,
            self.albedo()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optical_total_attenuation() {
        let props = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
        assert_eq!(props.total_attenuation(), 110.0);
    }

    #[test]
    fn test_optical_reduced_scattering() {
        let props = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
        assert!((props.reduced_scattering() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_optical_albedo() {
        let props = OpticalPropertyData::new(10.0, 90.0, 0.8, 1.4).unwrap();
        assert!((props.albedo() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_optical_mean_free_path() {
        let props = OpticalPropertyData::new(1.0, 99.0, 0.9, 1.4).unwrap();
        assert_eq!(props.mean_free_path(), 0.01);
    }

    #[test]
    fn test_optical_fresnel_reflectance() {
        let water = OpticalPropertyData::water();
        let reflectance = water.fresnel_reflectance_normal();
        assert!((reflectance - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_optical_validation() {
        assert!(OpticalPropertyData::new(-1.0, 100.0, 0.9, 1.4).is_err());
        assert!(OpticalPropertyData::new(10.0, 100.0, 1.5, 1.4).is_err());
        assert!(OpticalPropertyData::new(10.0, 100.0, 0.9, 0.5).is_err());
        assert!(OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).is_ok());
    }

    #[test]
    fn test_optical_presets() {
        let water = OpticalPropertyData::water();
        let tissue = OpticalPropertyData::soft_tissue();
        let blood = OpticalPropertyData::blood_oxygenated();

        assert!(water.absorption_coefficient < tissue.absorption_coefficient);
        assert!(blood.scattering_coefficient > water.scattering_coefficient);
        assert!(tissue.anisotropy > 0.8);
    }

    #[test]
    fn test_optical_penetration_depth() {
        let props = OpticalPropertyData::soft_tissue();
        let depth = props.penetration_depth();

        assert!(depth > 0.0);
        assert!(depth < 1.0);
    }
}
