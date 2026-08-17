//! CPMLConfig struct, Default, constructors, and computed methods.

use super::per_dimension::{PerDimensionAlpha, PerDimensionPML};
use kwavers_core::error::{ConfigError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Minimum cosine theta value to prevent division by zero in reflection estimation.
const MIN_COS_THETA_FOR_REFLECTION: f64 = 0.1;

/// Configuration for Convolutional PML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPMLConfig {
    /// Number of PML cells in each direction (legacy; use `per_dimension` instead).
    pub thickness: usize,
    /// Per-dimension PML thickness for k-Wave parity.
    pub per_dimension: PerDimensionPML,
    /// Polynomial order for profile grading (typically 3–4).
    pub polynomial_order: f64,
    /// Maximum conductivity scaling factor (uniform; overridden by
    /// `per_dimension_alpha`). This is k-Wave's `pml_alpha`, and the default of
    /// `2.0` is k-Wave's.
    ///
    /// On the convolutional FDTD path, `3.0` absorbs grazing incidence far
    /// better — 4.51e-5 of the incident peak against 1.23e-3 at 10 cells (27×),
    /// and 6.84e-9 against 4.26e-7 at 20 cells (62×), measured at 70° with σ-only
    /// CPML. Values above 3 give the optimum back. Prefer `3.0` for FDTD work
    /// that is not chasing k-Wave bit-parity.
    ///
    /// The default stays at k-Wave's `2.0` because this knob is shared with the
    /// split-field PSTD path, for which no equivalent measurement exists yet
    /// (KW-BND-100). Evidence:
    /// `sigma_factor_three_outperforms_the_kwave_default_at_grazing_incidence`.
    pub sigma_factor: f64,
    /// Per-dimension sigma scaling factors (k-Wave `pml_alpha` vector).
    pub per_dimension_alpha: PerDimensionAlpha,
    /// Maximum κ (real coordinate stretch) at the PML wall; `≥ 1`. `1` disables
    /// the κ term (σ-only CPML). Set via [`Self::with_cfs_pml`].
    ///
    /// The literature range is `≈ 5–20` (Komatitsch & Martin 2009), but measured
    /// against a reflection-free reference here, κ **degrades** a 10-cell PML at
    /// 70° incidence: 1.23e-3 of the direct peak at `κ_max = 1`, 1.77e-3 at `2`,
    /// 1.24e-2 at `10`. The q⁴ grading varies too steeply over 10 cells, and the
    /// discrete reflection it introduces exceeds what the stretch absorbs.
    ///
    /// Leave at `1` unless a thicker PML is in use and a measurement on the
    /// actual configuration shows a gain. Evidence:
    /// `cpml_absorbs_grazing_incidence_within_the_reflection_bound` in
    /// kwavers-solver; the earlier claim that `κ ≈ 5` was optimal was measured
    /// before KW-BND-097 fixed the staggered profile sampling, whose error
    /// dominated every κ comparison.
    pub kappa_max: f64,
    /// Maximum α (complex frequency shift) at the physical-domain interface; `≥ 0`.
    /// `0` disables the α term. CFS-PML uses `≈ π·f₀` to absorb evanescent and
    /// low-frequency/grazing energy (Roden & Gedney 2000). Set via [`Self::with_cfs_pml`].
    pub alpha_max: f64,
    /// Target reflection coefficient (e.g., 1e-6) used to *estimate* boundary
    /// performance via [`Self::theoretical_reflection`].
    ///
    /// It does **not** shape the absorbing profile. σ follows the k-Wave form
    /// `pml_alpha·(c/dx)·q⁴`, which carries no R₀ term, so tightening this
    /// value does not make the layer absorb more — raise `sigma_factor` or the
    /// PML thickness instead. Asserted by
    /// `target_reflection_does_not_enter_the_profile`; whether to adopt the
    /// Roden–Gedney `σ_max = −(m+1)c₀ln(R₀)/(2d)` grading in its place is
    /// tracked as KW-BND-099.
    pub target_reflection: f64,
    /// Enable grazing angle absorption.
    pub grazing_angle_absorption: bool,
    /// Suppress inner (low-index) z-PML cells for axisymmetric one-sided radial PML.
    pub radial_inner_z_transparent: bool,
}

impl Default for CPMLConfig {
    fn default() -> Self {
        let thickness = 20;
        Self {
            thickness,
            per_dimension: PerDimensionPML::uniform(thickness),
            polynomial_order: 3.0,
            sigma_factor: 2.0,
            per_dimension_alpha: PerDimensionAlpha::default(),
            // σ-only CPML by default (κ=1, α=0). These match the behavior that
            // was always effective: the prior 15.0/0.24 defaults were never read
            // by the profile kernel (dead config), and 0.24 is negligible vs the
            // physically-correct α_max ≈ π·f₀. CFS-PML is opt-in via with_cfs_pml.
            kappa_max: 1.0,
            alpha_max: 0.0,
            target_reflection: 1e-6,
            grazing_angle_absorption: true,
            radial_inner_z_transparent: false,
        }
    }
}

impl CPMLConfig {
    /// Create CPML configuration with uniform thickness.
    #[must_use]
    pub fn with_thickness(thickness: usize) -> Self {
        Self {
            thickness,
            per_dimension: PerDimensionPML::uniform(thickness),
            ..Default::default()
        }
    }

    /// Create CPML configuration with per-dimension thickness.
    #[must_use]
    pub fn with_per_dimension_thickness(x: usize, y: usize, z: usize) -> Self {
        let max_thickness = x.max(y).max(z);
        Self {
            thickness: max_thickness,
            per_dimension: PerDimensionPML::new(x, y, z),
            ..Default::default()
        }
    }

    /// Set per-dimension PML thickness (builder).
    #[must_use]
    pub fn with_pml_size(self, x: usize, y: usize, z: usize) -> Self {
        let max_thickness = x.max(y).max(z);
        Self {
            thickness: max_thickness,
            per_dimension: PerDimensionPML::new(x, y, z),
            ..self
        }
    }

    /// Set uniform sigma absorption factor (builder; k-Wave `pml_alpha` scalar).
    #[must_use]
    pub fn with_alpha(self, alpha: f64) -> Self {
        Self {
            sigma_factor: alpha,
            per_dimension_alpha: PerDimensionAlpha::uniform(alpha),
            ..self
        }
    }

    /// Enable complex-frequency-shifted PML (CFS-PML) with the given maxima (builder).
    ///
    /// Adds the graded real stretch `κ(q) = 1 + (κ_max−1)·q⁴` and frequency shift
    /// `α(q) = α_max·(1−q)` to the σ-only CPML, reducing spurious reflections at
    /// grazing incidence and for evanescent/low-frequency energy. This affects the
    /// convolutional (FDTD) boundary path; the k-Wave split-field (PSTD) decay
    /// factors derive from σ alone and are unchanged.
    ///
    /// Measured on a 10-cell PML at 70° incidence, neither term earns its place:
    /// α is neutral (1.21e-3 of the direct peak against 1.23e-3 for σ-only) and
    /// κ is harmful (see [`Self::kappa_max`]). The σ-only default is the
    /// recommended configuration at this thickness; reach for CFS only with a
    /// measurement on the actual geometry.
    /// `kappa_max = 1`, `alpha_max = 0` reduces exactly to
    /// σ-only CPML.
    ///
    /// # Panics
    /// Panics if `kappa_max < 1.0` or `alpha_max < 0.0` (invalid CFS parameters).
    #[must_use]
    pub fn with_cfs_pml(self, kappa_max: f64, alpha_max: f64) -> Self {
        assert!(
            kappa_max >= 1.0,
            "with_cfs_pml: kappa_max must be >= 1.0, got {kappa_max}"
        );
        assert!(
            alpha_max >= 0.0,
            "with_cfs_pml: alpha_max must be >= 0.0, got {alpha_max}"
        );
        Self {
            kappa_max,
            alpha_max,
            ..self
        }
    }

    /// Enable CFS-PML choosing `alpha_max = π·f₀` from the dominant source
    /// frequency `f₀` (builder).
    ///
    /// `α_max ≈ π·f₀` is the canonical complex-frequency-shift for the CFS-PML
    /// (Roden & Gedney 2000): it places the shift pole near the source band so
    /// evanescent and grazing-incidence energy is absorbed without reflecting
    /// propagating waves. Equivalent to
    /// `self.with_cfs_pml(kappa_max, std::f64::consts::PI * center_frequency_hz)`.
    ///
    /// # Panics
    /// Panics if `kappa_max < 1.0` or `center_frequency_hz < 0.0`.
    #[must_use]
    pub fn with_cfs_pml_for_frequency(self, kappa_max: f64, center_frequency_hz: f64) -> Self {
        assert!(
            center_frequency_hz >= 0.0,
            "with_cfs_pml_for_frequency: center_frequency_hz must be >= 0.0, got {center_frequency_hz}"
        );
        self.with_cfs_pml(kappa_max, std::f64::consts::PI * center_frequency_hz)
    }

    /// Suppress inner z-PML cells for one-sided axisymmetric radial PML (builder).
    #[must_use]
    pub fn with_radial_inner_z_transparent(self) -> Self {
        Self {
            radial_inner_z_transparent: true,
            ..self
        }
    }

    /// Set per-dimension sigma absorption factors (builder; k-Wave `pml_alpha` vector).
    #[must_use]
    pub fn with_alpha_xyz(self, ax: f64, ay: f64, az: f64) -> Self {
        let max_alpha = ax.max(ay).max(az);
        Self {
            sigma_factor: max_alpha,
            per_dimension_alpha: PerDimensionAlpha::new(ax, ay, az),
            ..self
        }
    }

    /// Sigma factor for a specific dimension (respects `per_dimension_alpha`).
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn sigma_factor_for_dimension(&self, dim: usize) -> KwaversResult<f64> {
        self.per_dimension_alpha.get(dim)
    }

    /// PML thickness for a specific dimension (0=x, 1=y, 2=z).
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn thickness_for_dimension(&self, dim: usize) -> KwaversResult<usize> {
        self.per_dimension.get(dim)
    }

    /// Validate all configuration parameters.
    /// # Errors
    /// - Returns [`kwavers_core::error::KwaversError::Config`] if any parameter violates its constraint.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thickness == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "thickness".to_owned(),
                value: self.thickness.to_string(),
                constraint: "CPML thickness must be positive".to_owned(),
            }
            .into());
        }

        if self.per_dimension.x == 0 && self.per_dimension.y == 0 && self.per_dimension.z == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "per_dimension".to_owned(),
                value: format!(
                    "({}, {}, {})",
                    self.per_dimension.x, self.per_dimension.y, self.per_dimension.z
                ),
                constraint: "At least one per-dimension PML thickness must be positive".to_owned(),
            }
            .into());
        }

        if self.polynomial_order < 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "polynomial_order".to_owned(),
                value: self.polynomial_order.to_string(),
                constraint: "Polynomial order must be >= 1.0".to_owned(),
            }
            .into());
        }

        if self.sigma_factor <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "sigma_factor".to_owned(),
                value: self.sigma_factor.to_string(),
                constraint: "Sigma factor must be positive".to_owned(),
            }
            .into());
        }

        if self.kappa_max < 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "kappa_max".to_owned(),
                value: self.kappa_max.to_string(),
                constraint: "Kappa max must be >= 1.0".to_owned(),
            }
            .into());
        }

        if self.alpha_max < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "alpha_max".to_owned(),
                value: self.alpha_max.to_string(),
                constraint: "Alpha max must be non-negative".to_owned(),
            }
            .into());
        }

        if self.target_reflection <= 0.0 || self.target_reflection >= 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "target_reflection".to_owned(),
                value: self.target_reflection.to_string(),
                constraint: "Target reflection must be between 0 and 1".to_owned(),
            }
            .into());
        }

        Ok(())
    }

    /// Theoretical reflection coefficient for a given grazing angle.
    ///
    /// Uses maximum thickness across all dimensions.
    /// Reference: Collino & Tsogka (2001).
    #[must_use]
    pub fn theoretical_reflection(&self, cos_theta: f64, dx: f64, sound_speed: f64) -> f64 {
        let cos_theta = cos_theta.max(MIN_COS_THETA_FOR_REFLECTION);
        let m = self.polynomial_order;
        let thickness = self.per_dimension.max_thickness() as f64;
        let sigma_max =
            self.sigma_factor * (m + 1.0) * sound_speed / (150.0 * std::f64::consts::PI * dx);
        self.target_reflection * (-(m + 1.0) * sigma_max * thickness * cos_theta).exp()
    }

    /// Theoretical reflection for a specific dimension.
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn theoretical_reflection_for_dimension(
        &self,
        dim: usize,
        cos_theta: f64,
        dx: f64,
        sound_speed: f64,
    ) -> KwaversResult<f64> {
        let cos_theta = cos_theta.max(MIN_COS_THETA_FOR_REFLECTION);
        let m = self.polynomial_order;
        let thickness = self.per_dimension.get(dim)? as f64;
        let sigma_max =
            self.sigma_factor * (m + 1.0) * sound_speed / (150.0 * std::f64::consts::PI * dx);
        Ok(self.target_reflection * (-(m + 1.0) * sigma_max * thickness * cos_theta).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn cfs_pml_for_frequency_sets_alpha_to_pi_f0() {
        let f0 = 1.5e6;
        let cfg = CPMLConfig::default().with_cfs_pml_for_frequency(10.0, f0);
        assert!(
            (cfg.alpha_max - PI * f0).abs() / (PI * f0) < 1e-12,
            "alpha_max {} should equal π·f₀ = {}",
            cfg.alpha_max,
            PI * f0
        );
        assert!((cfg.kappa_max - 10.0).abs() < 1e-12);
        cfg.validate().expect("π·f₀ CFS config must be valid");
    }

    #[test]
    fn cfs_pml_for_frequency_zero_reduces_to_sigma_only() {
        // f₀ = 0 ⇒ α_max = 0, i.e. σ-only CPML with the chosen κ.
        let cfg = CPMLConfig::default().with_cfs_pml_for_frequency(1.0, 0.0);
        assert_eq!(cfg.alpha_max, 0.0);
        assert_eq!(cfg.kappa_max, 1.0);
    }
}
