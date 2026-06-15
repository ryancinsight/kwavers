//! Volumetric (3-D) time-of-flight shear-wave-speed entry point (single-snapshot).
//!
//! Like [`super::time_of_flight`], a genuine 3-D time-of-flight estimate needs the
//! displacement **time series** to recover wavefront arrival times; the
//! single-snapshot [`DisplacementField`] cannot provide them. The previous body
//! used the same non-physical heuristic as the scalar path
//! (`arrival_time = distance/(|u|/max|u|·10)` ⟹ `c_s = 10·|u|/max|u|`, the distance
//! cancelling), so it has been removed and this delegates to the algebraic
//! Helmholtz inversion ([`super::direct::direct_inversion`]), the valid
//! single-snapshot estimator.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

/// Volumetric shear-wave-speed estimate from a single displacement snapshot.
///
/// Delegates to the algebraic Helmholtz inversion
/// [`super::direct::direct_inversion`] (single-snapshot data cannot support a true
/// time-of-flight estimate). See the module docs.
/// # Errors
/// - Propagates errors from [`super::direct::direct_inversion`].
pub(super) fn volumetric_time_of_flight_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    super::direct::direct_inversion(displacement, grid, density, frequency)
}
