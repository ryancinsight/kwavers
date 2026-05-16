//! Focused bowl element placement via golden-spiral (Fibonacci) sampling.

use super::super::geometry::Point3;
use crate::math::vector3::cross3;

/// Distribute `count` elements on a spherical-cap bowl.
///
/// The bowl is a spherical cap of radius `radius_m` centred at `focus_m`.
/// The cap vertex is `skin_contact_m` (the bowl vertex touching the skin).
/// Elements are placed at polar angles θ ∈ [θ_cutout, θ_max] using the
/// golden-spiral (Fibonacci) method for uniform area density.
///
/// Area-weighted sampling: the spherical-cap area element is `sin(θ) dθ dφ`.
/// Uniform area sampling requires the CDF `F(θ) = (cos(θ_cutout) − cos(θ)) /
/// (cos(θ_cutout) − cos(θ_max))` to be uniformly distributed, so
/// `cos(θ) = cos(θ_cutout) − t·(cos(θ_cutout) − cos(θ_max))`.
pub(super) fn bowl_elements(
    count: usize,
    skin_contact_m: Point3,
    focus_m: Point3,
    radius_m: f64,
) -> Vec<Point3> {
    let dx = focus_m.x_m - skin_contact_m.x_m;
    let dy = focus_m.y_m - skin_contact_m.y_m;
    let dz = focus_m.z_m - skin_contact_m.z_m;
    let dist = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

    if dist < 1.0e-6 || radius_m < 1.0e-6 {
        return vec![skin_contact_m; count];
    }

    // Bowl axis d̂ = (F − S) / ‖F − S‖.
    let axis = [dx / dist, dy / dist, dz / dist];
    let (e1, e2) = perpendicular_frame(axis);

    // Cap angular parameters.
    let theta_cutout: f64 = 0.175; // ≈ 10°
    let theta_max: f64 = 0.960; // ≈ 55°
    let cos_cutout = theta_cutout.cos();
    let cos_max = theta_max.cos();

    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

    (0..count)
        .map(|idx| {
            let t = (idx as f64 + 0.5) / count as f64;
            // Area-uniform cos(θ): linear interpolation on the cos(θ) axis.
            let cos_theta = cos_cutout - t * (cos_cutout - cos_max);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = idx as f64 * golden;

            // P = F − R·[cos(θ)·d̂ + sin(θ)·(cos(φ)·ê₁ + sin(φ)·ê₂)]
            Point3 {
                x_m: focus_m.x_m
                    - radius_m
                        * (cos_theta * axis[0]
                            + sin_theta * (phi.cos() * e1[0] + phi.sin() * e2[0])),
                y_m: focus_m.y_m
                    - radius_m
                        * (cos_theta * axis[1]
                            + sin_theta * (phi.cos() * e1[1] + phi.sin() * e2[1])),
                z_m: focus_m.z_m
                    - radius_m
                        * (cos_theta * axis[2]
                            + sin_theta * (phi.cos() * e1[2] + phi.sin() * e2[2])),
            }
        })
        .collect()
}

/// Construct an orthonormal frame (ê₁, ê₂) perpendicular to `axis`.
///
/// Uses the Gram–Schmidt process with a stable reference vector (not parallel
/// to `axis`).  Both output vectors are unit length.
fn perpendicular_frame(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let tmp = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1_raw = cross3(axis, tmp);
    let e1_len = (e1_raw[0] * e1_raw[0] + e1_raw[1] * e1_raw[1] + e1_raw[2] * e1_raw[2])
        .sqrt()
        .max(1.0e-12);
    let e1 = [e1_raw[0] / e1_len, e1_raw[1] / e1_len, e1_raw[2] / e1_len];
    let e2 = cross3(axis, e1);
    (e1, e2)
}
