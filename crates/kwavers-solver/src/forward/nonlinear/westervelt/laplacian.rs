//! Finite-difference Laplacian for the Westervelt FDTD solver.
//!
//! Operates in-place on the solver's pre-allocated `laplacian` workspace.
//! Supports 2nd-, 4th-, and 6th-order central-difference stencils. Unsupported
//! orders return a typed validation error.
//!
//! ## Stencil coefficients
//!
//! | Order | Coefficients | Truncation error |
//! |-------|--------------|-----------------|
//! | 2 | [1, −2, 1]/h² | O(h²) |
//! | 4 | [−1, 16, −30, 16, −1]/(12h²) | O(h⁴) |
//! | 6 | [2,−27,270,−490,270,−27,2]/(180h²) | O(h⁶) |
//!
//! **Theorem (exactness):** Any symmetric centered second-derivative stencil
//! with Σcₘ = 0 and Σm²·cₘ = 2 reproduces `d²(x²)/dx² = 2` exactly for all
//! interior points. Consequently `∇²(x²+y²+z²) = 6` exactly at all interior
//! nodes regardless of stencil order — used as the regression invariant in
//! `tests::westervelt_laplacian_stencils_are_exact_for_quadratic_fields`.
//!
//! ## Parallelism
//!
//! Outer `i` slabs of `laplacian` are disjoint; Moirai iterates them in
//! parallel while all workers share read-only access to `pressure`.

use kwavers_core::error::{GridError, KwaversError, KwaversResult, ValidationError};
use kwavers_grid::operators::coefficients::centered::{self, SecondDerivative};
use kwavers_grid::Grid;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

use super::WesterveltFdtd;

impl WesterveltFdtd {
    /// Compute `∇²p` into the pre-allocated `laplacian` workspace.
    ///
    /// Interior points are updated using the configured stencil order.
    /// Boundary ghost-layer entries (width = stencil radius) retain their
    /// previous values (zero after construction).
    ///
    /// Parallelism: outer `i` slabs of `laplacian` are disjoint and processed
    /// concurrently by Moirai; all workers share read-only access to `pressure`.
    ///
    /// # Errors
    /// Returns a grid dimension error if the supplied shape differs from the pressure field.
    /// Returns [`crate::KwaversError::Validation`] if `spatial_order` is not 2, 4, or 6.
    pub(super) fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
        let shape = [grid.nx, grid.ny, grid.nz];
        if self.pressure.shape() != shape {
            return Err(KwaversError::Grid(GridError::DimensionMismatch {
                expected: format!("{:?}", self.pressure.shape()),
                actual: format!("{shape:?}"),
            }));
        }
        match self.config.spatial_order {
            2 => self.write_laplacian(grid, &centered::SECOND),
            4 => self.write_laplacian(grid, &centered::FOURTH),
            6 => self.write_laplacian(grid, &centered::SIXTH),
            _ => {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "spatial_order".to_owned(),
                    value: self.config.spatial_order as f64,
                    reason: "Westervelt FDTD supports only 2, 4, or 6".to_owned(),
                }))
            }
        }
        Ok(())
    }

    fn write_laplacian<const R: usize>(&mut self, grid: &Grid, stencil: &SecondDerivative<R>) {
        let [nx, ny, nz] = [grid.nx, grid.ny, grid.nz];
        if nx <= 2 * R || ny <= 2 * R || nz <= 2 * R {
            return;
        }
        let slab_len = ny * nz;
        let inv_dx2 = 1.0 / (grid.dx * grid.dx);
        let inv_dy2 = 1.0 / (grid.dy * grid.dy);
        let inv_dz2 = 1.0 / (grid.dz * grid.dz);
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: standard-layout pressure");
        let laplacian = self
            .laplacian
            .as_slice_mut()
            .expect("invariant: standard-layout Laplacian");

        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(laplacian, slab_len, |i, slab| {
            if i < R || i >= nx - R {
                return;
            }
            for j in R..ny - R {
                for k in R..nz - R {
                    let local = j * nz + k;
                    let index = i * slab_len + local;
                    let along = |stride: usize| {
                        let p = pressure[index];
                        if R == 1 {
                            // Preserve the three-point stencil's rounding order.
                            return stencil.center.mul_add(p, pressure[index + stride])
                                + pressure[index - stride];
                        }
                        let mut sum = stencil.center * p;
                        for offset in 1..=R {
                            let delta = offset * stride;
                            let neighbors = pressure[index - delta] + pressure[index + delta];
                            sum = stencil.pairs[offset - 1].mul_add(neighbors, sum);
                        }
                        sum
                    };
                    slab[local] =
                        along(slab_len) * inv_dx2 + along(nz) * inv_dy2 + along(1) * inv_dz2;
                }
            }
        });
    }
}
