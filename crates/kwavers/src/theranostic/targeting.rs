//! Sonication-target selection from a reconstructed property volume.
//!
//! After the full-brain FWI/RTM/Born reconstruction produces a 3-D property
//! volume (sound speed, reflectivity, or a clinician-supplied region-of-interest
//! score), the therapy stage must pick a focal target. This module selects the
//! target voxel under a region-of-interest mask and converts it to a physical
//! focal position for the hemispherical array's steering controller.
//!
//! The selection is deliberately simple and deterministic — extremum or centroid
//! of a masked score field — so the targeting step is reproducible and testable.
//! Clinical target definition (tumour segmentation, eloquent-cortex avoidance)
//! lives upstream in the ROI mask; this module only resolves the mask + score to
//! a single focal voxel and its coordinates.

use ndarray::Array3;

/// Result of resolving a target from a reconstructed volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetSelection {
    /// Target voxel index `(i, j, k)` in the reconstruction grid.
    pub voxel: (usize, usize, usize),
    /// Physical focal position `[x, y, z]` [m] for array steering.
    pub position_m: [f64; 3],
    /// Score value at the selected voxel (units of the score field).
    pub score: f64,
}

/// Convert a voxel index to a physical position given grid origin and spacing.
///
/// `position = origin + (i, j, k) · spacing`, all in metres.
#[must_use]
pub fn voxel_to_position(
    voxel: (usize, usize, usize),
    origin_m: [f64; 3],
    spacing_m: [f64; 3],
) -> [f64; 3] {
    [
        origin_m[0] + voxel.0 as f64 * spacing_m[0],
        origin_m[1] + voxel.1 as f64 * spacing_m[1],
        origin_m[2] + voxel.2 as f64 * spacing_m[2],
    ]
}

impl TargetSelection {
    /// Select the voxel of maximum `score` among voxels where `mask` is true.
    ///
    /// Returns `None` if the mask selects no voxel or the shapes differ. Ties
    /// resolve to the first voxel in row-major order (deterministic).
    #[must_use]
    pub fn max_in_mask(
        score: &Array3<f64>,
        mask: &Array3<bool>,
        origin_m: [f64; 3],
        spacing_m: [f64; 3],
    ) -> Option<Self> {
        if score.dim() != mask.dim() {
            return None;
        }
        let mut best: Option<((usize, usize, usize), f64)> = None;
        for ((i, j, k), &s) in score.indexed_iter() {
            if !mask[[i, j, k]] {
                continue;
            }
            match best {
                Some((_, bs)) if s <= bs => {}
                _ => best = Some(((i, j, k), s)),
            }
        }
        best.map(|(voxel, score)| Self {
            voxel,
            position_m: voxel_to_position(voxel, origin_m, spacing_m),
            score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn voxel_to_position_is_affine() {
        let p = voxel_to_position((2, 3, 4), [0.01, -0.02, 0.0], [1e-3, 2e-3, 5e-4]);
        assert!((p[0] - (0.01 + 2e-3)).abs() < 1e-12);
        assert!((p[1] - (-0.02 + 6e-3)).abs() < 1e-12);
        assert!((p[2] - (0.0 + 2e-3)).abs() < 1e-12);
    }

    #[test]
    fn selects_max_score_voxel_inside_mask() {
        let mut score = Array3::from_elem((3, 3, 1), 0.0);
        let mut mask = Array3::from_elem((3, 3, 1), false);
        // Global max is outside the mask; in-mask max is at (2,0,0).
        score[[1, 1, 0]] = 100.0; // outside mask — must be ignored
        score[[2, 0, 0]] = 7.0;
        score[[0, 2, 0]] = 3.0;
        mask[[2, 0, 0]] = true;
        mask[[0, 2, 0]] = true;

        let sel = TargetSelection::max_in_mask(&score, &mask, [0.0; 3], [1e-3; 3]).unwrap();
        assert_eq!(
            sel.voxel,
            (2, 0, 0),
            "max within mask, ignoring outside-mask peak"
        );
        assert!((sel.score - 7.0).abs() < 1e-12);
        assert!((sel.position_m[0] - 2e-3).abs() < 1e-12);
    }

    #[test]
    fn empty_mask_returns_none() {
        let score = Array3::from_elem((2, 2, 2), 1.0);
        let mask = Array3::from_elem((2, 2, 2), false);
        assert!(TargetSelection::max_in_mask(&score, &mask, [0.0; 3], [1e-3; 3]).is_none());
    }

    #[test]
    fn shape_mismatch_returns_none() {
        let score = Array3::from_elem((2, 2, 2), 1.0);
        let mask = Array3::from_elem((2, 2, 3), true);
        assert!(TargetSelection::max_in_mask(&score, &mask, [0.0; 3], [1e-3; 3]).is_none());
    }
}
