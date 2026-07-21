//! Atlas parallel-provider adapters for medium field traversal.

use leto::Array3;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, for_each_chunk_mut_with, Adaptive};

pub(crate) const FIELD_CHUNK_SIZE: usize = 4096;

/// Fill a 3-D `array` with `f(x, y, z)` evaluated at each grid point in parallel.
///
/// Coordinate values are precomputed once and the flat row-major iteration is
/// chunked adaptively, avoiding the per-voxel overhead of `indices_to_coordinates`
/// and the sequential triple-nested loop.
pub(crate) fn fill_from_function<F>(array: &mut Array3<f64>, grid: &kwavers_grid::Grid, f: F)
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let [nx, ny, nz] = array.shape();
    let x: Vec<f64> = (0..nx)
        .map(|i| (i as f64).mul_add(grid.dx, grid.origin[0]))
        .collect();
    let y: Vec<f64> = (0..ny)
        .map(|j| (j as f64).mul_add(grid.dy, grid.origin[1]))
        .collect();
    let z: Vec<f64> = (0..nz)
        .map(|k| (k as f64).mul_add(grid.dz, grid.origin[2]))
        .collect();

    let slab = ny * nz;
    if let Some(values) = array.as_slice_mut() {
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            values,
            FIELD_CHUNK_SIZE,
            |chunk_index, chunk| {
                let base = chunk_index * FIELD_CHUNK_SIZE;
                for (lane, value) in chunk.iter_mut().enumerate() {
                    let flat = base + lane;
                    let i = flat / slab;
                    let rem = flat % slab;
                    let j = rem / nz;
                    let k = rem % nz;
                    *value = f(x[i], y[j], z[k]);
                }
            },
        );
    } else {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (gx, gy, gz) = grid.indices_to_coordinates(i, j, k);
                    array[[i, j, k]] = f(gx, gy, gz);
                }
            }
        }
    }
}

pub(crate) fn for_each_mut<T, F>(array: &mut Array3<T>, f: F)
where
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    if let Some(values) = array.as_slice_mut() {
        let f_ref = &f;
        for_each_chunk_mut_with::<Adaptive, _, _>(values, FIELD_CHUNK_SIZE, |chunk| {
            chunk.iter_mut().for_each(f_ref);
        });
    } else {
        let [nx, ny, nz] = array.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    f(&mut array[[i, j, k]]);
                }
            }
        }
    }
}

pub(crate) fn zip_mut_ref<T, U, F>(out: &mut Array3<T>, input: &Array3<U>, f: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    debug_assert_eq!(out.shape(), input.shape());
    match (out.as_slice_mut(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &input[base + lane]);
                    }
                },
            );
        }
        _ => {
            let [nx, ny, nz] = out.shape();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        f(&mut out[[i, j, k]], &input[[i, j, k]]);
                    }
                }
            }
        }
    }
}

pub(crate) fn zip_mut_two_refs<T, U, V, F>(
    out: &mut Array3<T>,
    first: &Array3<U>,
    second: &Array3<V>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    F: Fn(&mut T, &U, &V) + Send + Sync,
{
    debug_assert_eq!(out.shape(), first.shape());
    debug_assert_eq!(out.shape(), second.shape());
    match (out.as_slice_mut(), first.as_slice(), second.as_slice()) {
        (Some(out), Some(first), Some(second)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &first[base + lane], &second[base + lane]);
                    }
                },
            );
        }
        _ => {
            let [nx, ny, nz] = out.shape();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        f(&mut out[[i, j, k]], &first[[i, j, k]], &second[[i, j, k]]);
                    }
                }
            }
        }
    }
}

pub(crate) fn zip_mut_three_refs<T, U, V, W, F>(
    out: &mut Array3<T>,
    first: &Array3<U>,
    second: &Array3<V>,
    third: &Array3<W>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &U, &V, &W) + Send + Sync,
{
    debug_assert_eq!(out.shape(), first.shape());
    debug_assert_eq!(out.shape(), second.shape());
    debug_assert_eq!(out.shape(), third.shape());
    match (
        out.as_slice_mut(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
    ) {
        (Some(out), Some(first), Some(second), Some(third)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(
                            value,
                            &first[base + lane],
                            &second[base + lane],
                            &third[base + lane],
                        );
                    }
                },
            );
        }
        _ => {
            let [nx, ny, nz] = out.shape();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        f(
                            &mut out[[i, j, k]],
                            &first[[i, j, k]],
                            &second[[i, j, k]],
                            &third[[i, j, k]],
                        );
                    }
                }
            }
        }
    }
}
