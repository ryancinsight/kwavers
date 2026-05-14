//! 3-D spatial smoothing filters for elastography speed fields.
//!
//! References:
//! - Gonzalez & Woods (2008). *Digital Image Processing*, Ch. 5.
//! - Tomasi & Manduchi (1998). *Bilateral Filtering for Gray and Color Images*.
//! - Perona & Malik (1990). *Scale-space and edge detection using anisotropic diffusion*.

use ndarray::Array3;

/// Apply 3-D spatial smoothing (3×3×3 box filter) to reduce noise.
///
/// Each interior voxel is replaced by the mean of its 27-voxel neighbourhood.
/// Boundary voxels are unchanged.
pub fn spatial_smoothing(speed_field: &mut Array3<f64>) {
    let (nx, ny, nz) = speed_field.dim();
    let mut smoothed = speed_field.clone();

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let mut sum = 0.0;
                let mut count = 0;

                for di in -1..=1i32 {
                    for dj in -1..=1i32 {
                        for dk in -1..=1i32 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                sum += speed_field[[ii, jj, kk]];
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    smoothed[[i, j, k]] = sum / count as f64;
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
pub fn volumetric_smoothing(speed_field: &mut Array3<f64>) {
    let (nx, ny, nz) = speed_field.dim();
    let mut smoothed = speed_field.clone();

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let center = speed_field[[i, j, k]];
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for di in -1..=1i32 {
                    for dj in -1..=1i32 {
                        for dk in -1..=1i32 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                let neighbor = speed_field[[ii, jj, kk]];
                                let weight = (-(neighbor - center).abs()).exp();
                                sum += neighbor * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                }

                if weight_sum > 0.0 {
                    smoothed[[i, j, k]] = sum / weight_sum;
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
pub fn directional_smoothing(speed_field: &mut Array3<f64>) {
    let (nx, ny, nz) = speed_field.dim();
    let mut smoothed = speed_field.clone();

    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let center = speed_field[[i, j, k]];
                let x_dir = (speed_field[[i - 1, j, k]] + speed_field[[i + 1, j, k]]) / 2.0;
                let y_dir = (speed_field[[i, j - 1, k]] + speed_field[[i, j + 1, k]]) / 2.0;
                let z_dir = (speed_field[[i, j, k - 1]] + speed_field[[i, j, k + 1]]) / 2.0;

                smoothed[[i, j, k]] =
                    (center.mul_add(0.4, x_dir * 0.2) + y_dir * 0.2 + z_dir * 0.2).clamp(0.5, 10.0);
            }
        }
    }

    *speed_field = smoothed;
}
