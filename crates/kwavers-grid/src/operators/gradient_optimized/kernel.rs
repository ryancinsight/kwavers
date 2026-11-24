//! Axis-specialized centered differences for borrowed fields of any stride.
use super::BoundaryStrategy;
use crate::operators::{
    neighbors::{periodic, reflect},
    stencil::centered_first_derivative_sum,
};
use eunomia::FloatElement;
use leto::ArrayView3;
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

fn component<const AXIS: usize, T: FloatElement>(
    read: &impl Fn([usize; 3]) -> T,
    n: usize,
    p: [usize; 3],
    weights: &[T],
    scale: T,
    boundary: Option<BoundaryStrategy>,
) -> T {
    let i = p[AXIS];
    let sample = |at| {
        let mut pos = p;
        pos[AXIS] = at;
        read(pos)
    };
    if i >= weights.len() && n - i > weights.len() {
        return centered_first_derivative_sum(weights, |d| sample(i + d) - sample(i - d)) * scale;
    }
    let Some(boundary) = boundary else {
        return T::from_f64(0.0);
    };
    centered_first_derivative_sum(weights, |d| {
        let (left, right) = match boundary {
            BoundaryStrategy::ZeroPadding => {
                let lo = i.checked_sub(d).map_or(T::from_f64(0.0), sample);
                let hi = if d < n - i {
                    sample(i + d)
                } else {
                    T::from_f64(0.0)
                };
                return hi - lo;
            }
            BoundaryStrategy::Mirror => reflect(i, d, n),
            BoundaryStrategy::Periodic => periodic(i, d, n),
            BoundaryStrategy::Extrapolate => (i.saturating_sub(d), i.saturating_add(d).min(n - 1)),
        };
        sample(right) - sample(left)
    }) * scale
}

pub(super) fn fill<T: FloatElement>(
    field: &ArrayView3<T>,
    weights: &[T],
    scale: [T; 3],
    boundary: Option<BoundaryStrategy>,
    parallel_rows: Option<usize>,
    outputs: [&mut [T]; 3],
) {
    let shape @ [_, ny, nz] = field.shape();
    if let Some(data) = field.as_slice() {
        fill_rows(
            shape,
            weights,
            scale,
            boundary,
            parallel_rows,
            outputs,
            |[i, j, k]| data[(i * ny + j) * nz + k],
        );
    } else {
        fill_rows(
            shape,
            weights,
            scale,
            boundary,
            parallel_rows,
            outputs,
            |p| field[p],
        );
    }
}

fn fill_rows<T: FloatElement>(
    [nx, ny, nz]: [usize; 3],
    weights: &[T],
    scale: [T; 3],
    boundary: Option<BoundaryStrategy>,
    parallel_rows: Option<usize>,
    [x, y, z]: [&mut [T]; 3],
    read: impl Fn([usize; 3]) -> T + Sync,
) {
    if x.is_empty() {
        return;
    }
    let rows_per_chunk = parallel_rows.unwrap_or(1).clamp(1, x.len() / nz);
    let compute_rows = |first_row: usize, x: &mut [T], y: &mut [T], z: &mut [T]| {
        for (r, ((xs, ys), zs)) in x
            .chunks_exact_mut(nz)
            .zip(y.chunks_exact_mut(nz))
            .zip(z.chunks_exact_mut(nz))
            .enumerate()
        {
            let row = first_row + r;
            let (i, j) = (row / ny, row % ny);
            for k in 0..nz {
                let p = [i, j, k];
                xs[k] = component::<0, _>(&read, nx, p, weights, scale[0], boundary);
                ys[k] = component::<1, _>(&read, ny, p, weights, scale[1], boundary);
                zs[k] = component::<2, _>(&read, nz, p, weights, scale[2], boundary);
            }
        }
    };
    if parallel_rows.is_some() {
        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            x,
            y,
            z,
            rows_per_chunk * nz,
            |chunk, x, y, z| compute_rows(chunk * rows_per_chunk, x, y, z),
        );
    } else {
        compute_rows(0, x, y, z);
    }
}
