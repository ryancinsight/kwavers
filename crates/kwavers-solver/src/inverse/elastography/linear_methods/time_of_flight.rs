//! Time-of-flight shear-wave-speed entry point (single-snapshot).
//!
//! Genuine time-of-flight estimation requires the displacement **time series**
//! `u(x, t)` to track the shear wavefront's arrival time across space (Bercoff et
//! al. 2004, "Supersonic shear imaging"). The [`DisplacementField`] supplied here
//! carries a single spatial snapshot with no time axis, so an arrival-time
//! estimate is not recoverable from it.
//!
//! This entry therefore delegates to the algebraic Helmholtz inversion
//! ([`super::direct::direct_inversion`], McLaughlin & Renzi 2006), the
//! physically-valid single-snapshot estimator (solves `∇²u + k²u = 0` for
//! `k = ω/c_s` directly). The previous body fabricated an "arrival time" from the
//! displacement amplitude in which the propagation distance cancelled
//! algebraically (`c_s = 10·|u|/max|u|`), i.e. it computed no time-of-flight
//! physics; it has been removed.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

/// Estimate shear-wave speed from a single displacement snapshot.
///
/// Single-snapshot data cannot support a genuine arrival-time (time-of-flight)
/// estimate, so this delegates to the algebraic Helmholtz inversion
/// [`super::direct::direct_inversion`]. See the module docs.
/// # Errors
/// - Propagates errors from [`super::direct::direct_inversion`].
pub(super) fn time_of_flight_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    super::direct::direct_inversion(displacement, grid, density, frequency)
}
