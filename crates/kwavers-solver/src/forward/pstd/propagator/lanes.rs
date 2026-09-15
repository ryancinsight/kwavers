//! Z lanes of a C-order PSTD volume, scheduled as whole-lane moirai tasks.
//!
//! The split-field updates multiply each element by a factor that varies along
//! one axis only. Walking the volume lane by lane reads that factor once per
//! lane for x and y and zips it along the lane for z, where recovering
//! `(i, j, k)` from a flat index cost two divisions and two remainders per
//! element and kept the loop from vectorizing.

use moirai_parallel::{for_each_unit_task_mut_with, WorkBytes};

/// Bytes an element-wise pass moves before it spreads over workers.
///
/// The crossover apollo measured for its paired lane passes on the same
/// runtime and task width (`plan/fft/lanes.rs`, `PARALLEL_BYTES`: 32,768
/// complex elements, 512 KiB, apollo #442). The `pstd_long_run` bench is the
/// instrument that moves it for these kernels.
const LANE_PARALLEL_BYTES: usize = 512 * 1024;

/// Axis along which a spectral shift or PML factor varies.
#[derive(Clone, Copy)]
pub(super) enum LaneAxis {
    X,
    Y,
    Z,
}

/// The index into a one-axis factor table for element `(i, j, k)`.
#[inline]
pub(super) fn axis_index(axis: LaneAxis, i: usize, j: usize, k: usize) -> usize {
    match axis {
        LaneAxis::X => i,
        LaneAxis::Y => j,
        LaneAxis::Z => k,
    }
}

/// Runs `lane(start, i, j, values)` over every z lane of a C-order output whose
/// trailing extents are `[ny, nz]`: `start` is the lane's first flat index and
/// `values` its `nz` elements.
///
/// `element_bytes` counts one output element and every input element the
/// closure reads beside it, which is what a task moves.
pub(super) fn for_each_z_lane<T, F>(
    output: &mut [T],
    [ny, nz]: [usize; 2],
    element_bytes: usize,
    lane: F,
) where
    T: Send,
    F: Fn(usize, usize, usize, &mut [T]) + Send + Sync,
{
    for_each_unit_task_mut_with::<WorkBytes<LANE_PARALLEL_BYTES>, _, _, _, _>(
        output,
        nz,
        nz * element_bytes,
        || (),
        |(), first_lane, lanes| {
            for (offset, values) in lanes.chunks_exact_mut(nz).enumerate() {
                let index = first_lane + offset;
                lane(index * nz, index / ny, index % ny, values);
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::{for_each_z_lane, LANE_PARALLEL_BYTES};

    /// Every element of a `[nx, ny, nz]` volume receives the lane coordinates
    /// its flat index implies, whether the pass runs on one thread or spreads
    /// over tasks.
    fn assert_lane_coordinates(shape: [usize; 3], element_bytes: usize) {
        let [nx, ny, nz] = shape;
        let mut coordinates = vec![(usize::MAX, usize::MAX, usize::MAX); nx * ny * nz];
        for_each_z_lane(
            &mut coordinates,
            [ny, nz],
            element_bytes,
            |start, i, j, lane| {
                for (k, value) in lane.iter_mut().enumerate() {
                    assert_eq!(
                        *value,
                        (usize::MAX, usize::MAX, usize::MAX),
                        "an element is visited once"
                    );
                    *value = (i, j, start + k);
                }
            },
        );
        for (index, &(i, j, flat)) in coordinates.iter().enumerate() {
            assert_eq!((i, j, flat), (index / (ny * nz), (index / nz) % ny, index));
        }
    }

    #[test]
    fn lanes_carry_their_coordinates_serially() {
        let shape = [5, 3, 7];
        assert!(shape.iter().product::<usize>() * 8 < LANE_PARALLEL_BYTES);
        assert_lane_coordinates(shape, 8);
    }

    #[test]
    fn lanes_carry_their_coordinates_across_tasks() {
        let shape = [37, 29, 31];
        assert!(shape.iter().product::<usize>() * 24 >= LANE_PARALLEL_BYTES);
        assert_lane_coordinates(shape, 24);
    }
}
