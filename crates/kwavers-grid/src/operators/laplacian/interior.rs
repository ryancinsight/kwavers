//! Cartesian interior kernels. The last grid axis is contiguous.
use super::{LaplacianOperator, SecondDerivative};
use leto::{ArrayView3, ArrayViewMut3};
use moirai_parallel::{for_each_unit_task_mut_with, Adaptive};

impl LaplacianOperator {
    #[inline]
    pub(super) fn apply_second_order_interior(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
    ) {
        let [nx, ny, nz] = input.shape();
        if nx <= 2 || ny <= 2 || nz <= 2 {
            return;
        }

        if let Some(output) = output.as_mut_slice() {
            // One unit is an x-plane: it writes its own plane and the
            // stencil reads the plane either side of it, so a task carries
            // whole planes sized by the four it moves (moirai ADR 0059).
            let plane_len = ny * nz;
            let plane_bytes = plane_len.saturating_mul(4 * core::mem::size_of::<f64>());
            for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
                output,
                plane_len,
                plane_bytes,
                || (),
                |(), first_plane, planes| {
                    for (offset, plane) in planes.chunks_exact_mut(plane_len).enumerate() {
                        let i = first_plane + offset;
                        if i == 0 || i + 1 == nx {
                            continue;
                        }
                        for j in 1..ny - 1 {
                            let row_offset = j * nz;
                            for k in 1..nz - 1 {
                                plane[row_offset + k] = self.second_order_value(input, i, j, k);
                            }
                        }
                    }
                },
            );
            return;
        }

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    output[[i, j, k]] = self.second_order_value(input, i, j, k);
                }
            }
        }
    }

    #[inline]
    fn second_order_value(&self, input: ArrayView3<f64>, i: usize, j: usize, k: usize) -> f64 {
        let d2_dx2 = (2.0f64.mul_add(-input[[i, j, k]], input[[i + 1, j, k]])
            + input[[i - 1, j, k]])
            * self.dx2_inv;
        let d2_dy2 = (2.0f64.mul_add(-input[[i, j, k]], input[[i, j + 1, k]])
            + input[[i, j - 1, k]])
            * self.dy2_inv;
        let d2_dz2 = (2.0f64.mul_add(-input[[i, j, k]], input[[i, j, k + 1]])
            + input[[i, j, k - 1]])
            * self.dz2_inv;
        d2_dx2 + d2_dy2 + d2_dz2
    }

    pub(super) fn apply_interior<const R: usize>(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        stencil: &SecondDerivative<R>,
    ) {
        let [nx, ny, nz] = input.shape();
        if nx <= 2 * R || ny <= 2 * R || nz <= 2 * R {
            return;
        }
        if let (Some(src), Some(dst)) = (input.as_slice(), output.as_mut_slice()) {
            let plane = ny * nz;
            for i in R..nx - R {
                for j in R..ny - R {
                    let start = i * plane + j * nz + R;
                    for (k, result) in dst[start..start + nz - 2 * R].iter_mut().enumerate() {
                        let p = start + k;
                        *result = self.value(stencil, src[p], |d| {
                            [
                                src[p + d * plane] + src[p - d * plane],
                                src[p + d * nz] + src[p - d * nz],
                                src[p + d] + src[p - d],
                            ]
                        });
                    }
                }
            }
            return;
        }
        for i in R..nx - R {
            for j in R..ny - R {
                for k in R..nz - R {
                    output[[i, j, k]] = self.value(stencil, input[[i, j, k]], |d| {
                        [
                            input[[i + d, j, k]] + input[[i - d, j, k]],
                            input[[i, j + d, k]] + input[[i, j - d, k]],
                            input[[i, j, k + d]] + input[[i, j, k - d]],
                        ]
                    });
                }
            }
        }
    }
}
