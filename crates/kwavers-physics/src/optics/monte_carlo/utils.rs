use kwavers_core::constants::numerical::TWO_PI;
use crate::optics::monte_carlo::photon::Photon;
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};

/// Normalize vector
pub(crate) fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = v[2].mul_add(v[2], v[0].mul_add(v[0], v[1] * v[1])).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Cross product
pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -(a[2] * b[1])),
        a[2].mul_add(b[0], -(a[0] * b[2])),
        a[0].mul_add(b[1], -(a[1] * b[0])),
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
    let theta = TWO_PI * rng.gen::<f64>();
    let z = 2.0f64.mul_add(rng.gen::<f64>(), -1.0);
    let r = (1.0 - z * z).sqrt();
    [r * theta.cos(), r * theta.sin(), z]
}

/// Scatter photon using Henyey-Greenstein phase function
pub(crate) fn scatter_photon<R: Rng>(photon: &mut Photon, g: f64, rng: &mut R) {
    // Sample scattering angle using Henyey-Greenstein
    let cos_theta = if g.abs() < 1e-6 {
        // Isotropic scattering
        2.0f64.mul_add(rng.gen::<f64>(), -1.0)
    } else {
        let xi = rng.gen::<f64>();
        let temp = g.mul_add(-g, 1.0) / (2.0 * g).mul_add(xi, 1.0 - g);
        (g.mul_add(g, 1.0) - temp * temp) / (2.0 * g)
    };

    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TWO_PI * rng.gen::<f64>();

    // Get perpendicular basis
    let old_dir = photon.direction;
    let perp1 = get_perpendicular(old_dir);
    let perp2 = cross(old_dir, perp1);

    // New direction in local coordinates
    let new_dir = [
        (sin_theta * phi.cos()).mul_add(perp1[0], sin_theta * phi.sin() * perp2[0])
            + cos_theta * old_dir[0],
        (sin_theta * phi.cos()).mul_add(perp1[1], sin_theta * phi.sin() * perp2[1])
            + cos_theta * old_dir[1],
        (sin_theta * phi.cos()).mul_add(perp1[2], sin_theta * phi.sin() * perp2[2])
            + cos_theta * old_dir[2],
    ];

    photon.direction = normalize(new_dir);
}

/// Compute the distance from `pos` (inside voxel `(i, j, k)`) to the nearest
/// voxel-face boundary along direction `dir`, using the slab method.
///
/// ## Algorithm
///
/// For each axis a ∈ {x, y, z}:
///   - Low corner: `a0 = idx_a × da`.
///   - If `dir[a] > 0`: exit face at `a0 + da`, t_a = `(a0 + da − pos[a]) / dir[a]`.
///   - If `dir[a] < 0`: exit face at `a0`,        t_a = `(a0 − pos[a]) / dir[a]`.
///   - If `dir[a] ≈ 0`: t_a = ∞.
///
/// The photon exits the voxel at `t_min = min(t_x, t_y, t_z)`.
///
/// ## Returns
///
/// `(dist, next)`:
/// - `dist` — distance to the nearest voxel boundary (≥ 0).
/// - `next` — `Some((ni, nj, nk))` of the adjacent voxel, or `None` if the
///   adjacent voxel is outside the grid bounds.
///
/// ## Reference
///
/// Wang L, Jacques SL, Zheng L (1995). MCML §2.3.
/// Kay M, Kajiya J (1986). "Rendering complex scenes with memory-coherent
/// ray tracing." SIGGRAPH '86, §3.
#[allow(clippy::too_many_arguments)]
pub(crate) fn photon_step_to_boundary(
    pos: [f64; 3],
    dir: [f64; 3],
    i: usize,
    j: usize,
    k: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> (f64, Option<(usize, usize, usize)>) {
    const EPS: f64 = 1e-15;

    // Low corners of the current voxel
    let x0 = i as f64 * dx;
    let y0 = j as f64 * dy;
    let z0 = k as f64 * dz;

    // Parametric distance to exit face on each axis; also record traversal direction
    let (tx, x_pos) = if dir[0] > EPS {
        (((x0 + dx) - pos[0]) / dir[0], true)
    } else if dir[0] < -EPS {
        ((x0 - pos[0]) / dir[0], false)
    } else {
        (f64::INFINITY, true)
    };

    let (ty, y_pos) = if dir[1] > EPS {
        (((y0 + dy) - pos[1]) / dir[1], true)
    } else if dir[1] < -EPS {
        ((y0 - pos[1]) / dir[1], false)
    } else {
        (f64::INFINITY, true)
    };

    let (tz, z_pos) = if dir[2] > EPS {
        (((z0 + dz) - pos[2]) / dir[2], true)
    } else if dir[2] < -EPS {
        ((z0 - pos[2]) / dir[2], false)
    } else {
        (f64::INFINITY, true)
    };

    // Identify the first (minimum) boundary crossing
    let (t_min, axis, going_positive) = if tx <= ty && tx <= tz {
        (tx, 0usize, x_pos)
    } else if ty <= tz {
        (ty, 1usize, y_pos)
    } else {
        (tz, 2usize, z_pos)
    };

    // Compute the next voxel index along the crossing axis
    let next = match axis {
        0 => {
            if going_positive {
                let ni = i + 1;
                if ni < nx {
                    Some((ni, j, k))
                } else {
                    None
                }
            } else {
                i.checked_sub(1).map(|ni| (ni, j, k))
            }
        }
        1 => {
            if going_positive {
                let nj = j + 1;
                if nj < ny {
                    Some((i, nj, k))
                } else {
                    None
                }
            } else {
                j.checked_sub(1).map(|nj| (i, nj, k))
            }
        }
        _ => {
            if going_positive {
                let nk = k + 1;
                if nk < nz {
                    Some((i, j, nk))
                } else {
                    None
                }
            } else {
                k.checked_sub(1).map(|nk| (i, j, nk))
            }
        }
    };

    (t_min.max(0.0), next)
}

/// Atomic add for f64 (using bits as u64)
pub(crate) fn atomic_add(atomic: &AtomicU64, value: f64) {
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
