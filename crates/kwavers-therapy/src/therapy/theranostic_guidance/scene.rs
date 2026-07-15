//! CT-aligned transcranial target resolution.
//!
//! A transcranial scene stores target position as a normalized coordinate inside
//! the CT-derived foreground support.  Each simulation lattice resolves that
//! fraction against its own body or brain mask, so figure and solver paths use
//! the same anatomical target without sharing grid-specific voxel indices.

use leto::{Array2, Array3};

use kwavers_core::error::{KwaversError, KwaversResult};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds3 {
    min: [usize; 3],
    max: [usize; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds2 {
    min: [usize; 2],
    max: [usize; 2],
}

pub fn validate_target_fraction_xyz(fraction: [f64; 3]) -> KwaversResult<[f64; 3]> {
    if fraction
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(KwaversError::InvalidInput(
            "target_fraction_xyz values must be finite values in [0, 1]".to_owned(),
        ));
    }
    Ok(fraction)
}

pub fn target_index_from_mask_fraction_3d(
    mask: &Array3<bool>,
    fraction: [f64; 3],
) -> KwaversResult<[usize; 3]> {
    let fraction = validate_target_fraction_xyz(fraction)?;
    let bounds = bounds_3d(mask)?;
    Ok([
        fraction_to_index(bounds.min[0], bounds.max[0], fraction[0]),
        fraction_to_index(bounds.min[1], bounds.max[1], fraction[1]),
        fraction_to_index(bounds.min[2], bounds.max[2], fraction[2]),
    ])
}

pub(crate) fn target_index_from_mask_fraction_2d(
    mask: &Array2<bool>,
    fraction: [f64; 2],
) -> KwaversResult<(usize, usize)> {
    if fraction
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(KwaversError::InvalidInput(
            "target_fraction_xy values must be finite values in [0, 1]".to_owned(),
        ));
    }
    let bounds = bounds_2d(mask)?;
    Ok((
        fraction_to_index(bounds.min[0], bounds.max[0], fraction[0]),
        fraction_to_index(bounds.min[1], bounds.max[1], fraction[1]),
    ))
}

fn bounds_3d(mask: &Array3<bool>) -> KwaversResult<Bounds3> {
    let mut min = [usize::MAX; 3];
    let mut max = [0usize; 3];
    let mut any = false;
    for ([ix, iy, iz], active) in mask.indexed_iter() {
        if *active {
            min[0] = min[0].min(ix);
            min[1] = min[1].min(iy);
            min[2] = min[2].min(iz);
            max[0] = max[0].max(ix);
            max[1] = max[1].max(iy);
            max[2] = max[2].max(iz);
            any = true;
        }
    }
    any.then_some(Bounds3 { min, max })
        .ok_or_else(|| KwaversError::InvalidInput("target support mask is empty".to_owned()))
}

fn bounds_2d(mask: &Array2<bool>) -> KwaversResult<Bounds2> {
    let mut min = [usize::MAX; 2];
    let mut max = [0usize; 2];
    let mut any = false;
    for ([ix, iy], active) in mask.indexed_iter() {
        if *active {
            min[0] = min[0].min(ix);
            min[1] = min[1].min(iy);
            max[0] = max[0].max(ix);
            max[1] = max[1].max(iy);
            any = true;
        }
    }
    any.then_some(Bounds2 { min, max })
        .ok_or_else(|| KwaversError::InvalidInput("target support mask is empty".to_owned()))
}

fn fraction_to_index(min: usize, max: usize, fraction: f64) -> usize {
    let span = max.saturating_sub(min) as f64;
    (min as f64 + fraction * span).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_resolves_inside_3d_support_bounds() {
        let mut mask = Array3::<bool>::from_elem((9, 11, 13), false);
        for ix in 2..=6 {
            for iy in 3..=8 {
                for iz in 4..=10 {
                    mask[[ix, iy, iz]] = true;
                }
            }
        }

        let index = target_index_from_mask_fraction_3d(&mask, [0.5, 0.0, 1.0]).unwrap();

        assert_eq!(index, [4, 3, 10]);
    }

    #[test]
    fn fraction_rejects_empty_support() {
        let mask = Array3::<bool>::from_elem((3, 3, 3), false);

        let result = target_index_from_mask_fraction_3d(&mask, [0.5, 0.5, 0.5]);

        match result {
            Err(KwaversError::InvalidInput(message)) => {
                assert_eq!(message, "target support mask is empty");
            }
            other => panic!("expected empty-mask InvalidInput, got {other:?}"),
        }
    }
}
