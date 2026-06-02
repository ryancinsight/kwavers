//! Defaults for coupled bubble-field integration.

use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

/// Default liquid density used when no medium is supplied [kg m⁻³].
pub(super) const DEFAULT_RHO_LIQUID: f64 = DENSITY_WATER_NOMINAL;

/// Default physical grid spacing (m) (1 mm isotropic).
pub(super) const DEFAULT_GRID_SPACING: (f64, f64, f64) = (1e-3, 1e-3, 1e-3);

/// Ratio R_i / d_ij below which secondary Bjerknes coupling is negligible.
pub(super) const DEFAULT_COUPLING_THRESHOLD: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;

    /// Physical constants satisfy ordering invariants.
    #[test]
    fn constants_satisfy_ordering_invariants() {
        // Liquid density must be positive
        assert!(
            DEFAULT_RHO_LIQUID > 0.0,
            "DEFAULT_RHO_LIQUID must be positive, got {DEFAULT_RHO_LIQUID}"
        );
        // Grid spacings must be positive
        let (dx, dy, dz) = DEFAULT_GRID_SPACING;
        assert!(
            dx > 0.0 && dy > 0.0 && dz > 0.0,
            "grid spacings must be positive: ({dx}, {dy}, {dz})"
        );
        // Coupling threshold must be in (0, 1)
        assert!(
            DEFAULT_COUPLING_THRESHOLD > 0.0 && DEFAULT_COUPLING_THRESHOLD < 1.0,
            "coupling threshold {DEFAULT_COUPLING_THRESHOLD} must be in (0, 1)"
        );
    }
}
