//! Halo traversal and ghost-node mappings. The two z strips in an interior
//! row are disjoint even when the stencil covers the entire row.

use super::super::neighbors::{periodic, reflect};
use super::{LaplacianBoundary, LaplacianOperator, SecondDerivative};
use leto::{ArrayView3, ArrayViewMut3};
use std::ops::Range;

fn halo_rows<const R: usize>(
    [nx, ny, nz]: [usize; 3],
    mut visit: impl FnMut(usize, usize, Range<usize>),
) {
    for i in 0..nx {
        for j in 0..ny {
            if i < R || nx - i <= R || j < R || ny - j <= R || nz <= 2 * R {
                visit(i, j, 0..nz);
            } else {
                visit(i, j, 0..R);
                visit(i, j, nz - R..nz);
            }
        }
    }
}

impl LaplacianOperator {
    pub(super) fn apply_halo<const R: usize>(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        stencil: &SecondDerivative<R>,
    ) {
        match self.config.boundary {
            LaplacianBoundary::Dirichlet => {
                let shape @ [_, ny, nz] = input.shape();
                if let Some(dst) = output.as_mut_slice() {
                    halo_rows::<R>(shape, |i, j, range| {
                        let base = (i * ny + j) * nz;
                        dst[base + range.start..base + range.end].fill(0.0);
                    });
                } else {
                    halo_rows::<R>(shape, |i, j, range| {
                        for k in range {
                            output[[i, j, k]] = 0.0;
                        }
                    });
                }
            }
            LaplacianBoundary::Periodic => self.mapped_halo(input, output, stencil, periodic),
            LaplacianBoundary::Neumann => self.mapped_halo(input, output, stencil, reflect),
        }
    }

    fn mapped_halo<const R: usize>(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        stencil: &SecondDerivative<R>,
        neighbors: impl Fn(usize, usize, usize) -> (usize, usize),
    ) {
        let shape @ [nx, ny, nz] = input.shape();
        halo_rows::<R>(shape, |i, j, range| {
            for k in range {
                let p = [i, j, k];
                output[p] = self.value(stencil, input[p], |d| {
                    let (im, ip) = neighbors(i, d, nx);
                    let (jm, jp) = neighbors(j, d, ny);
                    let (km, kp) = neighbors(k, d, nz);
                    [
                        input[[ip, j, k]] + input[[im, j, k]],
                        input[[i, jp, k]] + input[[i, jm, k]],
                        input[[i, j, kp]] + input[[i, j, km]],
                    ]
                });
            }
        });
    }
}
