//! Defaults for coupled bubble-field integration.

/// Default liquid density used when no medium is supplied [kg m⁻³].
pub(super) const DEFAULT_RHO_LIQUID: f64 = 1000.0;

/// Default physical grid spacing [m] (1 mm isotropic).
pub(super) const DEFAULT_GRID_SPACING: (f64, f64, f64) = (1e-3, 1e-3, 1e-3);

/// Ratio R_i / d_ij below which secondary Bjerknes coupling is negligible.
pub(super) const DEFAULT_COUPLING_THRESHOLD: f64 = 0.01;
