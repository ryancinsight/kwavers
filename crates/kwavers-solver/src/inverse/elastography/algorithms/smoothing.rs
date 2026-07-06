//! 3-D spatial smoothing filters for elastography speed fields.
//!
//! References:
//! - Gonzalez & Woods (2008). *Digital Image Processing*, Ch. 5.
//! - Tomasi & Manduchi (1998). *Bilateral Filtering for Gray and Color Images*.
//! - Perona & Malik (1990). *Scale-space and edge detection using anisotropic diffusion*.

use super::volume::Volume3;

/// Apply 3-D spatial smoothing (3×3×3 box filter) to reduce noise.
///
/// Each interior voxel is replaced by the mean of its 27-voxel neighbourhood.
/// Boundary voxels are unchanged.
pub fn spatial_smoothing<V>(speed_field: &mut V)
where
    V: Volume3 + Clone,
{
    let (nx, ny, nz) = speed_field.dimensions();
    let mut smoothed = speed_field.clone();

    // Smooth all z-layers for a 2-D plane (nz < 3); the 27-voxel window is
    // bounds-checked below, so it clips correctly at a singleton z. Without this
    // the interior loop `1..nz-1` is empty for nz = 1 and no smoothing is applied
    // (the LFE/Helmholtz windowed averages then never form on 2-D input).
    let (k_lo, k_hi) = if nz >= 3 { (1, nz - 1) } else { (0, nz) };
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for k in k_lo..k_hi {
                let mut sum = 0.0;
                let mut count = 0;

                for di in -1..=1i32 {
                    for dj in -1..=1i32 {
                        for dk in -1..=1i32 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                sum += speed_field.value([ii, jj, kk]);
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    smoothed.set_value([i, j, k], sum / count as f64);
                }
            }
        }
    }

    *speed_field = smoothed;
}

/// Apply edge-preserving volumetric smoothing (bilateral-like filter).
///
/// Weights each neighbour by `exp(−|neighbour − center| / σ)` where `σ = 1`.
/// Large intensity gradients receive lower weights, preserving tissue boundaries.
pub fn volumetric_smoothing<V>(speed_field: &mut V)
where
    V: Volume3 + Clone,
{
    let (nx, ny, nz) = speed_field.dimensions();
    let mut smoothed = speed_field.clone();

    let (k_lo, k_hi) = if nz >= 3 { (1, nz - 1) } else { (0, nz) };
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for k in k_lo..k_hi {
                let center = speed_field.value([i, j, k]);
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for di in -1..=1i32 {
                    for dj in -1..=1i32 {
                        for dk in -1..=1i32 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                let neighbor = speed_field.value([ii, jj, kk]);
                                let weight = (-(neighbor - center).abs()).exp();
                                sum += neighbor * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                }

                if weight_sum > 0.0 {
                    smoothed.set_value([i, j, k], sum / weight_sum);
                }
            }
        }
    }

    *speed_field = smoothed;
}

/// Apply directional smoothing along coordinate axes.
///
/// Favours wave-propagation directions: 40 % centre + 20 % each axis.
/// Output is clamped to `[0.5, 10.0]` m/s.
pub fn directional_smoothing<V>(speed_field: &mut V)
where
    V: Volume3 + Clone,
{
    let (nx, ny, nz) = speed_field.dimensions();
    let mut smoothed = speed_field.clone();

    // For a 2-D plane (nz < 3) there is no z-neighbour pair; fold the z-weight
    // into the centre so the in-plane directional average still sums to one.
    let has_z_interior = nz >= 3;
    let (k_lo, k_hi) = if has_z_interior { (1, nz - 1) } else { (0, nz) };
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for k in k_lo..k_hi {
                let center = speed_field.value([i, j, k]);
                let x_dir =
                    (speed_field.value([i - 1, j, k]) + speed_field.value([i + 1, j, k])) / 2.0;
                let y_dir =
                    (speed_field.value([i, j - 1, k]) + speed_field.value([i, j + 1, k])) / 2.0;
                let (z_dir, z_w, c_w) = if has_z_interior {
                    (
                        (speed_field.value([i, j, k - 1]) + speed_field.value([i, j, k + 1])) / 2.0,
                        0.2,
                        0.4,
                    )
                } else {
                    (0.0, 0.0, 0.6)
                };

                smoothed.set_value(
                    [i, j, k],
                    (center.mul_add(c_w, x_dir * 0.2) + y_dir * 0.2 + z_dir * z_w).clamp(0.5, 10.0),
                );
            }
        }
    }

    *speed_field = smoothed;
}
