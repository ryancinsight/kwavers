//! Least-squares axial strain estimation.
//!
//! Thermal strain is the axial gradient of the apparent displacement,
//! `ε(z) = ∂u/∂z`. A naive finite difference amplifies tracking noise, so a
//! moving least-squares linear fit over an odd window is used: the strain at a
//! point is the slope of the best-fit line of `u` versus axial position over the
//! window. This is the Kallel–Ophir estimator.
//!
//! # Reference
//! - Kallel, F., & Ophir, J. (1997). "A least-squares strain estimator for
//!   elastography." *Ultrasonic Imaging*, 19(3), 195–208.

use leto::Array3;

/// Slope of the least-squares line through `values` sampled on a uniform grid
/// with spacing `dz`.
///
/// For samples `u_k` at positions `z_k = k·dz`, the slope estimate is
/// `Σ(z_k − z̄)(u_k − ū) / Σ(z_k − z̄)²`. The denominator depends only on the
/// (fixed, uniform) window geometry and is strictly positive for `len ≥ 2`.
fn least_squares_slope(values: &[f64], dz: f64) -> f64 {
    let n = values.len();
    debug_assert!(n >= 2, "slope requires at least two samples");
    let n_f = n as f64;
    let mean_z = (n_f - 1.0) / 2.0 * dz; // mean of 0,dz,...,(n-1)dz
    let mean_u = values.iter().sum::<f64>() / n_f;
    let mut num = 0.0;
    let mut den = 0.0;
    for (k, &u) in values.iter().enumerate() {
        let z = k as f64 * dz - mean_z;
        num += z * (u - mean_u);
        den += z * z;
    }
    num / den
}

/// Estimate the axial strain field from an apparent displacement field.
///
/// `displacement` is `[nx, ny, nz]` in metres with the axial direction along the
/// last axis; `dz` is the axial sample spacing in metres; `window` is the odd
/// least-squares window length (caller-validated). Near the axial boundaries,
/// where a full centered window does not fit, the window is truncated to the
/// available samples (still ≥ 2), so every depth receives an estimate.
#[must_use]
pub fn least_squares_strain(displacement: &Array3<f64>, dz: f64, window: usize) -> Array3<f64> {
    let (nx, ny, nz) = displacement.dim();
    let mut strain = Array3::zeros((nx, ny, nz));
    let half = window / 2;
    for i in 0..nx {
        for j in 0..ny {
            for z in 0..nz {
                let lo = z.saturating_sub(half);
                let hi = (z + half + 1).min(nz);
                if hi - lo < 2 {
                    continue;
                }
                let win: Vec<f64> = (lo..hi).map(|k| displacement[[i, j, k]]).collect();
                strain[[i, j, z]] = least_squares_slope(&win, dz);
            }
        }
    }
    strain
}
