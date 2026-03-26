use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::physics::optics::monte_carlo::photon::Photon;

/// Normalize vector
pub(crate) fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Cross product
pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Get perpendicular vector
pub(crate) fn get_perpendicular(v: [f64; 3]) -> [f64; 3] {
    let abs_x = v[0].abs();
    let abs_y = v[1].abs();
    let abs_z = v[2].abs();

    if abs_x < abs_y && abs_x < abs_z {
        normalize(cross(v, [1.0, 0.0, 0.0]))
    } else if abs_y < abs_z {
        normalize(cross(v, [0.0, 1.0, 0.0]))
    } else {
        normalize(cross(v, [0.0, 0.0, 1.0]))
    }
}

/// Sample isotropic direction
pub(crate) fn sample_isotropic_direction<R: Rng>(rng: &mut R) -> [f64; 3] {
    let theta = 2.0 * std::f64::consts::PI * rng.gen::<f64>();
    let z = 2.0 * rng.gen::<f64>() - 1.0;
    let r = (1.0 - z * z).sqrt();
    [r * theta.cos(), r * theta.sin(), z]
}

/// Scatter photon using Henyey-Greenstein phase function
pub(crate) fn scatter_photon<R: Rng>(photon: &mut Photon, g: f64, rng: &mut R) {
    // Sample scattering angle using Henyey-Greenstein
    let cos_theta = if g.abs() < 1e-6 {
        // Isotropic scattering
        2.0 * rng.gen::<f64>() - 1.0
    } else {
        let xi = rng.gen::<f64>();
        let temp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi);
        (1.0 + g * g - temp * temp) / (2.0 * g)
    };

    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * std::f64::consts::PI * rng.gen::<f64>();

    // Get perpendicular basis
    let old_dir = photon.direction;
    let perp1 = get_perpendicular(old_dir);
    let perp2 = cross(old_dir, perp1);

    // New direction in local coordinates
    let new_dir = [
        sin_theta * phi.cos() * perp1[0]
            + sin_theta * phi.sin() * perp2[0]
            + cos_theta * old_dir[0],
        sin_theta * phi.cos() * perp1[1]
            + sin_theta * phi.sin() * perp2[1]
            + cos_theta * old_dir[1],
        sin_theta * phi.cos() * perp1[2]
            + sin_theta * phi.sin() * perp2[2]
            + cos_theta * old_dir[2],
    ];

    photon.direction = normalize(new_dir);
}

/// Atomic add for f64 (using bits as u64)
pub(crate) fn atomic_add_f64(atomic: &AtomicU64, value: f64) {
    let mut old = atomic.load(Ordering::Relaxed);
    loop {
        let old_f64 = f64::from_bits(old);
        let new_f64 = old_f64 + value;
        let new = new_f64.to_bits();

        match atomic.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => old = x,
        }
    }
}
