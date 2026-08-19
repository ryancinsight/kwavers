//! Axial displacement tracking by normalized cross-correlation (NCC).
//!
//! Estimates the apparent axial shift between a pre-heating ("reference") and a
//! post-heating ("tracked") RF/echo volume. For each lateral position and depth
//! the integer lag that maximizes the windowed NCC is found, then refined to
//! sub-sample precision by parabolic interpolation of the correlation peak.
//!
//! The implementation delegates NCC computation and parabolic sub-sample
//! refinement to `ritk-block-matching` (atlas US-023-D3), which is the SSOT
//! for these operations across the stack. The kwavers 1-D A-line tracking case
//! maps cleanly onto the 3-D seam with `dims=[1,1,nz]` and radii on the fast
//! (axial) axis only.
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
///
/// Delegates NCC and parabolic sub-sample refinement to `ritk-block-matching`
/// (atlas US-023-D3 — the SSOT for these operations across the stack).
/// The 1-D tracking case maps onto the 3-D seam as `dims=[1,1,nz]` with
/// radii on the fast (axial) axis only.
#[must_use]
pub fn track_line_samples(
    reference: ArrayView1<f64>,
    tracked: ArrayView1<f64>,
    params: TrackingParams,
) -> Vec<f64> {
    let nz = reference.size();
    let guard = params.window_half + params.max_lag;
    let mut disp = vec![0.0; nz];
    if nz <= 2 * guard {
        return disp;
    }

    // Build flat row-major [1, 1, nz] buffers for the block-matching seam.
    let fixed: Vec<f64> = reference.iter().copied().collect();
    let moving: Vec<f64> = tracked.iter().copied().collect();
    let dims = [1usize, 1, nz];
    let config = BlockMatchingConfig {
        block_radius: [0, 0, params.window_half],
        search_radius: [0, 0, params.max_lag],
    };

    for z in guard..(nz - guard) {
        let result = match_block(
            &fixed,
            &moving,
            dims,
            [0, 0, z],
            config,
            SubpixelRefinement::Parabolic,
        );
        if let Ok(bd) = result {
            disp[z] = bd.displacement[2];
        }
        // On error (e.g. search boundary): leave 0.0, matching prior behaviour.
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
