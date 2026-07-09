//! Axial displacement tracking by normalized cross-correlation (NCC).
//!
//! Estimates the apparent axial shift between a pre-heating ("reference") and a
//! post-heating ("tracked") RF/echo volume. For each lateral position and depth
//! the integer lag that maximizes the windowed NCC is found, then refined to
//! sub-sample precision by parabolic interpolation of the correlation peak.
//!
//! The convention is: a positive displacement means the post-heating echo
//! appears at a larger axial index (later round-trip time, i.e. apparently
//! farther from the transducer). Displacements are returned in metres using the
//! axial sample spacing `Δz = c₀ / (2 f_s)`.
//!
//! # References
//! - Pinton, G. F., Dahl, J. J., & Trahey, G. E. (2006). "Rapid tracking of
//!   small displacements with ultrasound." *IEEE TUFFC*, 53(6), 1103–1117.
//! - Lubinski, M. A., Emelianov, S. Y., & O'Donnell, M. (1999). "Speckle
//!   tracking methods for ultrasonic elasticity imaging using short-time
//!   correlation." *IEEE TUFFC*, 46(1), 82–96.

use leto::{
    Array3,
    ArrayView1,
};

/// Parameters controlling the cross-correlation displacement estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingParams {
    /// Correlation kernel half-length in axial samples (kernel length `2w+1`).
    pub window_half: usize,
    /// Maximum search lag in axial samples (`|displacement| ≤ max_lag`).
    pub max_lag: usize,
}

impl Default for TrackingParams {
    fn default() -> Self {
        Self {
            window_half: 8,
            max_lag: 6,
        }
    }
}

/// Normalized cross-correlation of two equal-length windows.
///
/// Returns a value in `[-1, 1]`. Zero is returned when either window has no
/// variance (constant signal), for which displacement is undefined.
fn ncc(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let da = va - mean_a;
        let db = vb - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

/// Sub-sample peak location from three correlation samples by fitting a
/// parabola through `(−1, c_m1), (0, c_0), (1, c_p1)`.
///
/// Returns the offset in `[−0.5, 0.5]` of the true peak relative to the integer
/// maximum. Falls back to `0.0` when the curvature is non-negative (flat or
/// degenerate peak).
fn parabolic_subsample(c_m1: f64, c_0: f64, c_p1: f64) -> f64 {
    let denom = c_m1 - 2.0 * c_0 + c_p1;
    if denom.abs() < f64::EPSILON || denom >= 0.0 {
        return 0.0;
    }
    let delta = 0.5 * (c_m1 - c_p1) / denom;
    // Guard against ill-conditioned fits driving the estimate outside one sample.
    delta.clamp(-0.5, 0.5)
}

/// Estimate the apparent axial displacement of a single RF line, in samples.
///
/// `reference[z]` is matched against `tracked[z + lag]` over `lag ∈
/// [−max_lag, max_lag]`. Entries within `window_half + max_lag` of either end,
/// where the kernel or search window would leave the array, are left at `0.0`.
#[must_use]
pub fn track_line_samples(
    reference: ArrayView1<f64>,
    tracked: ArrayView1<f64>,
    params: TrackingParams,
) -> Vec<f64> {
    let nz = reference.len();
    let mut disp = vec![0.0; nz];
    let w = params.window_half;
    let max_lag = params.max_lag as isize;
    let guard = w + params.max_lag;
    if nz <= 2 * guard {
        return disp;
    }
    // `z` is the window centre, used in slice arithmetic (z±w, z+lag) as well as
    // to store the result; it is not a plain element index.
    #[allow(clippy::needless_range_loop)]
    for z in guard..(nz - guard) {
        let ref_win = referenceslice(&[(Some((z - w) as isize) as usize, Some(=(z + w) as isize) as usize, 1)]);
        let mut best_corr = f64::NEG_INFINITY;
        let mut best_lag = 0isize;
        // Sample correlation at every integer lag, retaining the three values
        // around the maximum for sub-sample refinement.
        let mut corr = vec![0.0; (2 * max_lag + 1) as usize];
        for (idx, lag) in (-max_lag..=max_lag).enumerate() {
            let center = (z as isize + lag) as usize;
            let trk_win = trackedslice(&[(Some((center - w) as isize) as usize, Some(=(center + w) as isize) as usize, 1)]);
            let c = ncc(ref_win, trk_win);
            corr[idx] = c;
            if c > best_corr {
                best_corr = c;
                best_lag = lag;
            }
        }
        let best_idx = (best_lag + max_lag) as usize;
        let sub = if best_idx == 0 || best_idx == corr.len() - 1 {
            0.0
        } else {
            parabolic_subsample(corr[best_idx - 1], corr[best_idx], corr[best_idx + 1])
        };
        disp[z] = best_lag as f64 + sub;
    }
    disp
}

/// Estimate the apparent axial displacement field (m) for a full volume.
///
/// `reference` and `tracked` are `[nx, ny, nz]` RF volumes with the axial
/// (fast-time) direction along the last axis. `dz` is the axial sample spacing
/// in metres (see [`super::ThermalStrainConfig::axial_sample_spacing`]).
#[must_use]
pub fn track_axial_displacement(
    reference: &Array3<f64>,
    tracked: &Array3<f64>,
    params: TrackingParams,
    dz: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = reference.dim();
    let mut field = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            let ref_line = reference.slice_with::<1>(&[SliceArg::Index(i as isize), SliceArg::Index(j as isize), SliceArg::All]);
            let trk_line = tracked.slice_with::<1>(&[SliceArg::Index(i as isize), SliceArg::Index(j as isize), SliceArg::All]);
            let disp_samples = track_line_samples(ref_line, trk_line, params);
            for (z, &d) in disp_samples.iter().enumerate() {
                field[[i, j, z]] = d * dz;
            }
        }
    }
    field
}
