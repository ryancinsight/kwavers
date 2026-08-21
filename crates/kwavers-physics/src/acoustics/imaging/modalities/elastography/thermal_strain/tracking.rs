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
//! # Where the matching lives
//!
//! The correlation search and its sub-sample peak estimate are not implemented
//! here: they are `ritk_block_matching`, the stack's single block-matching
//! seam. This module owns the *ultrasound* part — the guard band, the axial
//! sample spacing, and the displacement sign convention — and delegates the
//! matching itself, so there is one NCC and one parabolic-peak implementation
//! in the stack rather than two that can drift.
//!
//! # References
//! - Pinton, G. F., Dahl, J. J., & Trahey, G. E. (2006). "Rapid tracking of
//!   small displacements with ultrasound." *IEEE TUFFC*, 53(6), 1103–1117.
//! - Lubinski, M. A., Emelianov, S. Y., & O'Donnell, M. (1999). "Speckle
//!   tracking methods for ultrasonic elasticity imaging using short-time
//!   correlation." *IEEE TUFFC*, 46(1), 82–96.

use leto::{Array3, ArrayView1, SliceArg};
use ritk_block_matching::{match_block, BlockMatchingConfig, SubpixelRefinement};

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
    let nz = reference.size();
    let mut disp = vec![0.0; nz];
    let w = params.window_half;
    let guard = w + params.max_lag;
    if nz <= 2 * guard {
        return disp;
    }

    // The seam works on flat buffers; a line is the degenerate 3-D case with
    // the axial direction on the fast axis.
    let reference: Vec<f64> = reference.iter().copied().collect();
    let tracked: Vec<f64> = tracked.iter().copied().collect();
    let dims = [1, 1, nz];
    let config = BlockMatchingConfig {
        block_radius: [0, 0, w],
        search_radius: [0, 0, params.max_lag],
    };

    for (z, displacement) in disp.iter_mut().enumerate().take(nz - guard).skip(guard) {
        // A featureless window has no defined displacement. The seam says so by
        // returning an error rather than a fabricated peak; here that is the
        // long-standing "leave it at zero" behaviour for a window the estimator
        // cannot speak about.
        *displacement = match_block(
            &reference,
            &tracked,
            dims,
            [0, 0, z],
            config,
            SubpixelRefinement::Parabolic,
        )
        .map_or(0.0, |result| result.displacement[2]);
    }
    disp
}

/// Estimate the apparent axial displacement field (m) for a full volume.
///
/// `reference` and `tracked` are `[nx, ny, nz]` RF volumes with the axial
/// (fast-time) direction along the last axis. `dz` is the axial sample spacing
/// in metres (see [`super::ThermalStrainConfig::axial_sample_spacing`]).
///
/// # Panics
///
/// Panics if either input cannot provide the indexed axial slice implied by
/// `reference`'s dimensions, or if the two volumes have incompatible shapes.
#[must_use]
pub fn track_axial_displacement(
    reference: &Array3<f64>,
    tracked: &Array3<f64>,
    params: TrackingParams,
    dz: f64,
) -> Array3<f64> {
    let [nx, ny, nz] = reference.shape();
    let mut field = Array3::zeros([nx, ny, nz]);
    for i in 0..nx {
        for j in 0..ny {
            let ref_line = reference
                .slice_with::<1>(&[
                    SliceArg::Index(i as isize),
                    SliceArg::Index(j as isize),
                    SliceArg::All,
                ])
                .expect("reference slice is in bounds");
            let trk_line = tracked
                .slice_with::<1>(&[
                    SliceArg::Index(i as isize),
                    SliceArg::Index(j as isize),
                    SliceArg::All,
                ])
                .expect("tracked slice is in bounds");
            let disp_samples = track_line_samples(ref_line, trk_line, params);
            for (z, &d) in disp_samples.iter().enumerate() {
                field[[i, j, z]] = d * dz;
            }
        }
    }
    field
}
