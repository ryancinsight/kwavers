//! Isotropic phase-gradient shear-wave-speed entry point (single-snapshot).
//!
//! A correct phase-gradient estimator recovers the local wavenumber from the
//! log-derivative of a harmonic displacement, `|∇u|/|u| ≈ |k|`, then `c_s = ω/k`
//! (this is exactly what [`super::directional`] does, dividing by the **local**
//! `|u|`). The previous body here instead divided the displacement gradient by the
//! **global maximum** amplitude, so it did not estimate a local wavenumber and the
//! result was not a physically-meaningful speed; it has been removed.
//!
//! This isotropic entry delegates to the algebraic Helmholtz inversion
//! ([`super::direct::direct_inversion`], McLaughlin & Renzi 2006), which uses the
//! full `∇²u + k²u = 0` relation rather than a single-direction gradient. For the
//! per-axis local-wavenumber phase-gradient method, use
//! [`super::directional::directional_phase_gradient_inversion`].

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

/// Isotropic phase-gradient shear-wave-speed estimate from a single snapshot.
///
/// Delegates to the algebraic Helmholtz inversion
/// [`super::direct::direct_inversion`] (McLaughlin & Renzi 2006). See the module
/// docs for why the previous global-max-normalised gradient was removed.
/// # Errors
/// - Propagates errors from [`super::direct::direct_inversion`].
pub(super) fn phase_gradient_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    super::direct::direct_inversion(displacement, grid, density, frequency)
}
