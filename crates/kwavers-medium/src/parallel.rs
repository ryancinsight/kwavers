//! Atlas parallel-provider adapters for medium field traversal.

use leto::Array3;
use moirai_parallel::{for_each_unit_task_mut_with, Adaptive};

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
        // One unit writes one value; the three axis tables it reads are
        // shared and small, so the width comes from the output element.
        for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
            values,
            1,
            core::mem::size_of::<f64>(),
            || (),
            |(), first, run| {
                for (lane, value) in run.iter_mut().enumerate() {
                    let flat = first + lane;
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
        // One unit is read and written in place, so it moves twice its size.
        for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
            values,
            1,
            2 * core::mem::size_of::<T>(),
            || (),
            |(), _, run| run.iter_mut().for_each(f_ref),
        );
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
            // One unit is read and written in `out` and read once in the
            // input, which is what sizes the task.
            for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
                out,
                1,
                2 * core::mem::size_of::<T>() + core::mem::size_of::<U>(),
                || (),
                |(), first, run| {
                    for (lane, value) in run.iter_mut().enumerate() {
                        f_ref(value, &input[first + lane]);
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
            // One unit is read and written in `out` and read once in each
            // input, which is what sizes the task.
            for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
                out,
                1,
                2 * core::mem::size_of::<T>()
                    + core::mem::size_of::<U>()
                    + core::mem::size_of::<V>(),
                || (),
                |(), first_unit, run| {
                    for (lane, value) in run.iter_mut().enumerate() {
                        f_ref(value, &first[first_unit + lane], &second[first_unit + lane]);
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
            // One unit is read and written in `out` and read once in each of
            // the three inputs, which is what sizes the task.
            for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
                out,
                1,
                2 * core::mem::size_of::<T>()
                    + core::mem::size_of::<U>()
                    + core::mem::size_of::<V>()
                    + core::mem::size_of::<W>(),
                || (),
                |(), first_unit, run| {
                    for (lane, value) in run.iter_mut().enumerate() {
                        f_ref(
                            value,
                            &first[first_unit + lane],
                            &second[first_unit + lane],
                            &third[first_unit + lane],
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

#[cfg(test)]
mod tests {
    use super::{
        fill_from_function, for_each_mut, zip_mut_ref, zip_mut_three_refs, zip_mut_two_refs,
    };
    use leto::Array3;

    /// A shape past the policy's parallel floor and several unit tasks wide,
    /// so a task boundary that dropped, doubled or shifted a run shows up.
    const SHAPE: [usize; 3] = [32, 32, 32];

    /// `index / 4` as an exactly representable value, distinct per element.
    fn ramp() -> Array3<f64> {
        let mut value = 0.0;
        Array3::from_shape_vec(
            SHAPE,
            (0..SHAPE[0] * SHAPE[1] * SHAPE[2])
                .map(|_| {
                    value += 0.25;
                    value
                })
                .collect(),
        )
        .expect("shape matches the data")
    }

    #[test]
    fn each_cell_is_filled_from_its_own_coordinates() {
        let grid = kwavers_grid::Grid::new(SHAPE[0], SHAPE[1], SHAPE[2], 1.0e-3, 2.0e-3, 5.0e-4)
            .expect("invariant: the test grid dimensions and spacings are valid");
        let mut field = Array3::zeros(SHAPE);

        fill_from_function(&mut field, &grid, |x, y, z| {
            z.mul_add(3.0, x.mul_add(1.0, y * 2.0))
        });

        for i in 0..SHAPE[0] {
            for j in 0..SHAPE[1] {
                for k in 0..SHAPE[2] {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let expected = z.mul_add(3.0, x.mul_add(1.0, y * 2.0));
                    assert_eq!(
                        field[[i, j, k]].to_bits(),
                        expected.to_bits(),
                        "fill[{i}, {j}, {k}]"
                    );
                }
            }
        }
    }

    #[test]
    fn every_cell_is_visited_once_in_place() {
        let source = ramp();
        let mut field = source.clone();

        for_each_mut(&mut field, |value| *value = value.mul_add(-2.0, 0.25));

        for (index, (got, before)) in field
            .as_slice()
            .unwrap()
            .iter()
            .zip(source.as_slice().unwrap())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                before.mul_add(-2.0, 0.25).to_bits(),
                "in place[{index}]"
            );
        }
    }

    #[test]
    fn every_cell_pairs_with_its_own_inputs() {
        let input = ramp();
        let second = ramp();
        let third = ramp();

        let mut one = Array3::zeros(SHAPE);
        zip_mut_ref(&mut one, &input, |slot, &value| *slot = value * 3.0);

        let mut two = Array3::zeros(SHAPE);
        zip_mut_two_refs(&mut two, &input, &second, |slot, &a, &b| {
            *slot = a.mul_add(2.0, b)
        });

        let mut three = Array3::zeros(SHAPE);
        zip_mut_three_refs(&mut three, &input, &second, &third, |slot, &a, &b, &c| {
            *slot = c.mul_add(4.0, a.mul_add(2.0, b));
        });

        let values = input.as_slice().unwrap();
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                one.as_slice().unwrap()[index].to_bits(),
                (value * 3.0).to_bits(),
                "one input[{index}]"
            );
            assert_eq!(
                two.as_slice().unwrap()[index].to_bits(),
                value.mul_add(2.0, *value).to_bits(),
                "two inputs[{index}]"
            );
            assert_eq!(
                three.as_slice().unwrap()[index].to_bits(),
                value.mul_add(4.0, value.mul_add(2.0, *value)).to_bits(),
                "three inputs[{index}]"
            );
        }
    }
}
