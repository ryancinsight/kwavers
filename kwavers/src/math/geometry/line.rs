use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Create a line mask connecting two points using a Bresenham-like 3D algorithm.
pub fn make_line(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    start: [f64; 3],
    end: [f64; 3],
) -> KwaversResult<Array3<bool>> {
    let (nx, ny, nz) = dim;
    let (dx, dy, dz) = spacing;

    let mut mask = Array3::from_elem((nx, ny, nz), false);

    let start_i = (start[0] / dx).round() as isize;
    let start_j = (start[1] / dy).round() as isize;
    let start_k = (start[2] / dz).round() as isize;

    let end_i = (end[0] / dx).round() as isize;
    let end_j = (end[1] / dy).round() as isize;
    let end_k = (end[2] / dz).round() as isize;

    let di = (end_i - start_i).abs();
    let dj = (end_j - start_j).abs();
    let dk = (end_k - start_k).abs();

    let si = if end_i > start_i { 1 } else { -1 };
    let sj = if end_j > start_j { 1 } else { -1 };
    let sk = if end_k > start_k { 1 } else { -1 };

    let dm = di.max(dj).max(dk);

    let mut i = start_i;
    let mut j = start_j;
    let mut k = start_k;

    let mut ei = dm / 2;
    let mut ej = dm / 2;
    let mut ek = dm / 2;

    for _ in 0..=dm {
        if i >= 0 && i < nx as isize && j >= 0 && j < ny as isize && k >= 0 && k < nz as isize {
            mask[[i as usize, j as usize, k as usize]] = true;
        }

        ei -= di;
        if ei < 0 {
            i += si;
            ei += dm;
        }

        ej -= dj;
        if ej < 0 {
            j += sj;
            ej += dm;
        }

        ek -= dk;
        if ek < 0 {
            k += sk;
            ek += dm;
        }
    }

    Ok(mask)
}
