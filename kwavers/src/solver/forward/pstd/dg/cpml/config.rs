//! Per-axis CPML configuration for the tensor acoustic DG solver.
//!
//! Holds the Roden-Gedney grading parameters used by [`super::profiles::DgCpmlProfiles`]
//! to generate σ(ξ), κ(ξ), α(ξ) along each Cartesian axis. One [`DgCpmlAxis`] is
//! carried per active axis; inactive (length-1) axes ignore their entry.
//!
//! # Reference
//! Roden, J. A., & Gedney, S. D. (2000). Convolutional PML (CPML): An
//! efficient FDTD implementation of the CFS-PML for arbitrary media.
//! *Microwave & Optical Technology Letters*, 27(5), 334–339.

use crate::core::error::{KwaversError, KwaversResult};

/// Per-axis CPML grading parameters.
///
/// Layer thickness is expressed in elements (not nodes) so the absorbing zone
/// always aligns with the DG element grid. A thickness of `t` elements maps to
/// `t * n_nodes` GLL nodes along the axis.
///
/// # Theorem (Roden-Gedney 2000, Eq. 8)
/// For target reflection `R₀` (typical 1e-6), polynomial grading order `m`,
/// reference speed `c₀`, and layer depth `d = thickness_elements · element_span`:
///
/// ```text
/// σ_max = -(m + 1) · c₀ · ln(R₀) / (2 · d)
/// σ(ξ) = σ_max · (ξ/d)^m,   ξ ∈ [0, d] measured from the inner PML face
/// κ(ξ) = 1 + (κ_max − 1) · (ξ/d)^m
/// α(ξ) = α_max · (1 − ξ/d)
/// ```
///
/// The Cartesian field on `[0, d]` is mirrored at the opposite end of the
/// domain (left + right strips). Inner physical-domain nodes carry σ = 0,
/// κ = 1, α = 0, which reduces the CPML auxiliary ODE to `dΨ/dt = 0`, leaving
/// the standard DG RHS unchanged.
#[derive(Debug, Clone, Copy)]
pub struct DgCpmlAxis {
    /// PML thickness in elements. `0` disables CPML on this axis.
    pub thickness_elements: usize,
    /// Target reflection coefficient `R₀` for normal incidence (typical `1e-6`).
    pub target_reflection: f64,
    /// Polynomial grading order `m` for σ(ξ) (Roden-Gedney recommend `m ∈ [3, 4]`).
    pub polynomial_order: u32,
    /// Maximum κ stretching value (`1.0` reduces CFS-PML to classical CPML).
    pub kappa_max: f64,
    /// Maximum complex-frequency-shift `α_max` [rad/s] (`0.0` disables CFS).
    pub alpha_max: f64,
}

impl DgCpmlAxis {
    /// Disabled axis: zero thickness, no absorption.
    pub const DISABLED: Self = Self {
        thickness_elements: 0,
        target_reflection: 1.0,
        polynomial_order: 4,
        kappa_max: 1.0,
        alpha_max: 0.0,
    };

    /// Standard 10-element layer with `R₀ = 10⁻⁶`, `m = 4`, no CFS.
    #[must_use]
    pub fn standard(thickness_elements: usize) -> Self {
        Self {
            thickness_elements,
            target_reflection: 1.0e-6,
            polynomial_order: 4,
            kappa_max: 1.0,
            alpha_max: 0.0,
        }
    }

    /// Returns `true` when this axis has a non-empty absorbing layer.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.thickness_elements > 0
    }

    /// Validate axis parameters.
    ///
    /// # Errors
    /// Returns an error when grading parameters are physically invalid.
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.is_active() {
            return Ok(());
        }
        if !self.target_reflection.is_finite()
            || self.target_reflection <= 0.0
            || self.target_reflection >= 1.0
        {
            return Err(KwaversError::InvalidInput(format!(
                "DgCpmlAxis.target_reflection must lie in (0, 1), got {}",
                self.target_reflection
            )));
        }
        if self.polynomial_order < 1 {
            return Err(KwaversError::InvalidInput(
                "DgCpmlAxis.polynomial_order must be >= 1".to_owned(),
            ));
        }
        if !self.kappa_max.is_finite() || self.kappa_max < 1.0 {
            return Err(KwaversError::InvalidInput(format!(
                "DgCpmlAxis.kappa_max must be >= 1, got {}",
                self.kappa_max
            )));
        }
        if !self.alpha_max.is_finite() || self.alpha_max < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "DgCpmlAxis.alpha_max must be finite and non-negative, got {}",
                self.alpha_max
            )));
        }
        Ok(())
    }
}

/// Full per-axis `[x, y, z]` CPML configuration for the tensor acoustic DG solver.
///
/// One axis with `thickness_elements = 0` keeps that direction untouched (e.g.,
/// a slab simulation may use CPML on `x` and `y` while leaving `z` periodic).
#[derive(Debug, Clone, Copy)]
pub struct DgCpmlConfig {
    /// Per-axis grading: `[x, y, z]`.
    pub axes: [DgCpmlAxis; 3],
}

impl DgCpmlConfig {
    /// Symmetric layer of the same standard thickness on all three axes.
    #[must_use]
    pub fn uniform(thickness_elements: usize) -> Self {
        Self {
            axes: [DgCpmlAxis::standard(thickness_elements); 3],
        }
    }

    /// Per-axis convenience constructor.
    #[must_use]
    pub fn with_axes(x: DgCpmlAxis, y: DgCpmlAxis, z: DgCpmlAxis) -> Self {
        Self { axes: [x, y, z] }
    }

    /// Validate all configured axes.
    ///
    /// # Errors
    /// Returns an error when any active axis violates its grading invariants.
    pub fn validate(&self) -> KwaversResult<()> {
        for axis in &self.axes {
            axis.validate()?;
        }
        Ok(())
    }

    /// Returns `true` when any axis carries a non-empty PML.
    #[must_use]
    pub fn any_active(&self) -> bool {
        self.axes.iter().any(DgCpmlAxis::is_active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_axis_validates_silently() {
        assert!(DgCpmlAxis::DISABLED.validate().is_ok());
        assert!(!DgCpmlAxis::DISABLED.is_active());
    }

    #[test]
    fn standard_layer_is_active_and_valid() {
        let axis = DgCpmlAxis::standard(10);
        assert!(axis.is_active());
        assert!(axis.validate().is_ok());
        assert_eq!(axis.thickness_elements, 10);
        assert_eq!(axis.polynomial_order, 4);
        assert_eq!(axis.target_reflection, 1.0e-6);
    }

    #[test]
    fn target_reflection_outside_open_unit_interval_is_rejected() {
        let mut axis = DgCpmlAxis::standard(8);
        axis.target_reflection = 0.0;
        assert!(axis.validate().is_err());
        axis.target_reflection = 1.0;
        assert!(axis.validate().is_err());
        axis.target_reflection = -0.1;
        assert!(axis.validate().is_err());
        axis.target_reflection = f64::NAN;
        assert!(axis.validate().is_err());
    }

    #[test]
    fn kappa_max_below_one_is_rejected() {
        let mut axis = DgCpmlAxis::standard(8);
        axis.kappa_max = 0.5;
        assert!(axis.validate().is_err());
    }

    #[test]
    fn negative_alpha_max_is_rejected() {
        let mut axis = DgCpmlAxis::standard(8);
        axis.alpha_max = -1.0;
        assert!(axis.validate().is_err());
    }

    #[test]
    fn uniform_config_activates_all_axes() {
        let cfg = DgCpmlConfig::uniform(10);
        assert!(cfg.any_active());
        assert!(cfg.validate().is_ok());
        for axis in &cfg.axes {
            assert_eq!(axis.thickness_elements, 10);
        }
    }

    #[test]
    fn with_axes_zero_thickness_disables_independently() {
        let cfg = DgCpmlConfig::with_axes(
            DgCpmlAxis::standard(10),
            DgCpmlAxis::standard(10),
            DgCpmlAxis::DISABLED,
        );
        assert!(cfg.any_active());
        assert!(!cfg.axes[2].is_active());
    }
}
