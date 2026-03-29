//! PSTD Source Injection Mode Classification
//!
//! Determines how a source mask should be injected into the PSTD solver —
//! as a boundary plane (unit scale) or as a normalised point/volume source.
//!
//! # Theorem — Source Scale Invariance (Treeby & Cox 2010, §2.4)
//!
//! For additive pressure sources in a pseudospectral solver the source term
//! must be normalised by the number of active grid points so that the total
//! injected energy is independent of source geometry (point, line, plane).
//!
//! Formally: if the source occupies N active voxels, each voxel receives an
//! amplitude scaled by 1/N so that Σᵢ mask[i] × amplitude = amplitude.
//!
//! Exception: a boundary-plane source (all active points share the same index
//! on one axis and that axis is at the domain boundary) injects a plane wave
//! at unit amplitude, because the plane is the natural "source boundary" in
//! k-space methods. No normalisation is applied in this case.
//!
//! # References
//!
//! Treeby, B. E., & Cox, B. T. (2010). k-Wave: MATLAB toolbox for the
//! simulation and reconstruction of photoacoustic wave fields.
//! *Journal of Biomedical Optics*, 15(2), 021314.
//! <https://doi.org/10.1117/1.3360308>

use crate::domain::source::SourceInjectionMode;
use ndarray::Array3;
use tracing::debug;

/// Classify how `mask` should be injected during PSTD time-stepping.
///
/// ## Algorithm
///
/// 1. Collect all active (|mask| > ε) voxel indices.
/// 2. Determine whether all active voxels share the same index on one axis
///    **and** that index is on the domain boundary (0 or N−1).
/// 3. If so, return `Additive { scale: 1.0 }` (boundary plane — no normalisation).
/// 4. Otherwise return `Additive { scale: 1/N }` where N = number of active voxels.
///
/// For PSTD the always-returned variant is `Additive`; `Boundary` is not used
/// because spectral methods cannot enforce true Dirichlet boundary conditions.
///
/// ## Arguments
/// * `mask` — Source spatial mask (active voxels have |mask| > 1e-12).
#[must_use]
pub fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
    let shape = mask.dim();
    let mut num_active: usize = 0;
    let mut first_i: Option<usize> = None;
    let mut first_j: Option<usize> = None;
    let mut first_k: Option<usize> = None;
    let mut all_same_i = true;
    let mut all_same_j = true;
    let mut all_same_k = true;
    let mut mask_sum = 0.0_f64;
    let mut mask_min = f64::MAX;
    let mut mask_max = f64::MIN;

    for ((i, j, k), &m) in mask.indexed_iter() {
        if m.abs() > 1e-12 {
            num_active += 1;
            mask_sum += m;
            mask_min = mask_min.min(m);
            mask_max = mask_max.max(m);

            match first_i {
                Some(fi) if fi != i => all_same_i = false,
                None => first_i = Some(i),
                _ => {}
            }
            match first_j {
                Some(fj) if fj != j => all_same_j = false,
                None => first_j = Some(j),
                _ => {}
            }
            match first_k {
                Some(fk) if fk != k => all_same_k = false,
                None => first_k = Some(k),
                _ => {}
            }
        }
    }

    // A boundary plane has all active voxels at the same index on one axis,
    // and that index is exactly 0 or N−1 for that axis.
    let is_boundary_plane = (all_same_i
        && (first_i == Some(0) || first_i == Some(shape.0 - 1)))
        || (all_same_j && (first_j == Some(0) || first_j == Some(shape.1 - 1)))
        || (all_same_k && (first_k == Some(0) || first_k == Some(shape.2 - 1)));

    let scale = if is_boundary_plane {
        1.0 // Plane wave: each boundary point receives full amplitude
    } else if num_active > 0 {
        1.0 / (num_active as f64) // Point/volume: normalise over active voxels
    } else {
        1.0 // Empty mask — scale is irrelevant
    };

    debug!(
        num_active,
        mask_sum,
        mask_min,
        mask_max,
        is_boundary_plane,
        scale,
        "PSTD source injection mode determined"
    );
    debug!(
        "  Mask geometry: all_same_i={}, all_same_j={}, all_same_k={}, \
         first_i={:?}, first_j={:?}, first_k={:?}",
        all_same_i, all_same_j, all_same_k, first_i, first_j, first_k
    );

    SourceInjectionMode::Additive { scale }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Single active voxel in the interior → normalised (scale = 1.0 / 1 = 1.0).
    #[test]
    fn test_single_interior_point_scale_one() {
        let mut mask = Array3::<f64>::zeros((8, 8, 8));
        mask[[4, 4, 4]] = 1.0;
        let mode = determine_injection_mode(&mask);
        match mode {
            SourceInjectionMode::Additive { scale } => {
                assert!((scale - 1.0).abs() < 1e-12, "scale = {scale}");
            }
            _ => panic!("Expected Additive"),
        }
    }

    /// Four active voxels in the interior → scale = 0.25.
    #[test]
    fn test_four_interior_points_normalised() {
        let mut mask = Array3::<f64>::zeros((8, 8, 8));
        mask[[2, 2, 2]] = 1.0;
        mask[[3, 2, 2]] = 1.0;
        mask[[2, 3, 2]] = 1.0;
        mask[[2, 2, 3]] = 1.0;
        let mode = determine_injection_mode(&mask);
        match mode {
            SourceInjectionMode::Additive { scale } => {
                assert!((scale - 0.25).abs() < 1e-12, "scale = {scale}");
            }
            _ => panic!("Expected Additive"),
        }
    }

    /// Full x=0 plane → boundary plane, scale = 1.0.
    #[test]
    fn test_boundary_plane_x0_unit_scale() {
        let mut mask = Array3::<f64>::zeros((8, 4, 4));
        for j in 0..4 {
            for k in 0..4 {
                mask[[0, j, k]] = 1.0;
            }
        }
        let mode = determine_injection_mode(&mask);
        match mode {
            SourceInjectionMode::Additive { scale } => {
                assert!(
                    (scale - 1.0).abs() < 1e-12,
                    "boundary plane must have scale=1, got {scale}"
                );
            }
            _ => panic!("Expected Additive"),
        }
    }

    /// Empty mask → scale = 1.0 (irrelevant, no injection occurs).
    #[test]
    fn test_empty_mask_scale_one() {
        let mask = Array3::<f64>::zeros((4, 4, 4));
        let mode = determine_injection_mode(&mask);
        match mode {
            SourceInjectionMode::Additive { scale } => {
                assert!((scale - 1.0).abs() < 1e-12, "scale = {scale}");
            }
            _ => panic!("Expected Additive"),
        }
    }
}
