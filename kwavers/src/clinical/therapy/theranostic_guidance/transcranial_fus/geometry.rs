use std::f64::consts::PI;

use ndarray::Array2;

/// Generate hemispherical element positions using a Fibonacci spiral.
///
/// # Arguments
/// * `element_count` — number of elements.
/// * `radius_m` — hemisphere radius.
/// * `cap_min_polar_rad` — minimum polar angle (co-latitude from +z).
/// * `cap_max_polar_rad` — maximum polar angle.
///
/// # Returns
/// `(element_count, 3)` array of [x, y, z] positions in metres.
pub fn fibonacci_hemisphere_positions(
    element_count: usize,
    radius_m: f64,
    cap_min_polar_rad: f64,
    cap_max_polar_rad: f64,
) -> Array2<f64> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let cos_min = cap_min_polar_rad.cos();
    let cos_max = cap_max_polar_rad.cos();
    Array2::from_shape_fn((element_count, 3), |(idx, col)| {
        let t = (idx as f64 + 0.5) / element_count as f64;
        let cos_theta = cos_min + (cos_max - cos_min) * t;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = golden * idx as f64;
        match col {
            0 => radius_m * sin_theta * phi.cos(),
            1 => radius_m * sin_theta * phi.sin(),
            _ => -radius_m * cos_theta,
        }
    })
}
