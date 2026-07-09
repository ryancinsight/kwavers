use leto::Array2;

use kwavers_core::error::KwaversResult;
use kwavers_transducer::transducers::focused::{SphericalCapConfig, SphericalCapLayout};

/// Generate focused spherical-cap element positions.
///
/// # Arguments
/// * `element_count` — number of elements.
/// * `radius_m` — cap radius from the acoustic focus.
/// * `cap_min_polar_rad` — minimum polar angle (co-latitude from +z).
/// * `cap_max_polar_rad` — maximum polar angle.
///
/// # Returns
/// `(element_count, 3)` array of [x, y, z] positions in metres.
///
/// # Errors
///
/// Returns an error when count, radius, axis, or polar span is outside the
/// focused spherical-cap domain.
pub fn focused_cap_positions(
    element_count: usize,
    radius_m: f64,
    cap_min_polar_rad: f64,
    cap_max_polar_rad: f64,
) -> KwaversResult<Array2<f64>> {
    let layout = SphericalCapLayout::new(SphericalCapConfig::focused_cap(
        element_count,
        radius_m,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        cap_min_polar_rad,
        cap_max_polar_rad,
    ))?;

    Ok(Array2::from_shape_fn((element_count, 3), |(idx, col)| {
        layout.elements()[idx].position_m[col]
    }))
}
