use ndarray::{Array2, Array3};

use crate::core::error::{KwaversError, KwaversResult};

/// Generate GBM subspot raster positions within a tumour mask.
///
/// Iterates the tumour bounding box at a stride derived from `pitch_m` /
/// `spacing_m`, keeps all interior voxels, appends the centroid, deduplicates.
///
/// # Returns
/// `(M, 3)` array of voxel indices `[ix, iy, iz]`.
pub fn gbm_subspot_raster(
    tumor_mask: &Array3<bool>,
    spacing_m: [f64; 3],
    pitch_m: f64,
) -> KwaversResult<Array2<usize>> {
    let (nx, ny, nz) = tumor_mask.dim();

    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    let mut count = 0_usize;
    let mut lo = [nx, ny, nz];
    let mut hi = [0_usize; 3];
    for ((ix, iy, iz), &active) in tumor_mask.indexed_iter() {
        if active {
            cx += ix as f64;
            cy += iy as f64;
            cz += iz as f64;
            count += 1;
            lo[0] = lo[0].min(ix);
            lo[1] = lo[1].min(iy);
            lo[2] = lo[2].min(iz);
            hi[0] = hi[0].max(ix);
            hi[1] = hi[1].max(iy);
            hi[2] = hi[2].max(iz);
        }
    }
    if count == 0 {
        return Err(KwaversError::InvalidInput(
            "tumor mask contains no active voxels".to_owned(),
        ));
    }
    let centroid = [
        (cx / count as f64).round() as usize,
        (cy / count as f64).round() as usize,
        (cz / count as f64).round() as usize,
    ];

    let stride = [
        (pitch_m / spacing_m[0]).round() as usize,
        (pitch_m / spacing_m[1]).round() as usize,
        (pitch_m / spacing_m[2]).round() as usize,
    ];
    let stride = [stride[0].max(1), stride[1].max(1), stride[2].max(1)];

    let mut candidates: Vec<[usize; 3]> = Vec::new();
    let mut ix = lo[0];
    while ix <= hi[0] {
        let mut iy = lo[1];
        while iy <= hi[1] {
            let mut iz = lo[2];
            while iz <= hi[2] {
                if tumor_mask[[ix, iy, iz]] {
                    candidates.push([ix, iy, iz]);
                }
                iz += stride[2];
            }
            iy += stride[1];
        }
        ix += stride[0];
    }

    let centroid_in = tumor_mask[[centroid[0], centroid[1], centroid[2]]];
    if !centroid_in || !candidates.contains(&centroid) {
        candidates.push(centroid);
    }

    candidates.sort_unstable();
    candidates.dedup();

    if candidates.is_empty() {
        candidates.push(centroid);
    }

    let m = candidates.len();
    let arr = Array2::from_shape_fn((m, 3), |(row, col)| candidates[row][col]);
    Ok(arr)
}
