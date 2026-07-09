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

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
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
    /// Returns [`KwaversError::Validation`] if `spatial_order` is not 2, 4, or 6.
    pub(super) fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let slab_len = ny * nz;
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: Westervelt pressure is standard-layout");
        let laplacian = self
            .laplacian
            .as_slice_mut()
            .expect("invariant: Westervelt laplacian is standard-layout");
        let index = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;

        match self.config.spatial_order {
            2 => {
                for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                    laplacian,
                    slab_len,
                    |i, lap_i| {
                        if i == 0 || i == nx - 1 {
                            return;
                        }
                        for j in 1..ny - 1 {
                            for k in 1..nz - 1 {
                                let center = index(i, j, k);
                                let p = pressure[center];
                                lap_i[j * nz + k] = (2.0f64
                                    .mul_add(-p, pressure[index(i + 1, j, k)])
                                    + pressure[index(i - 1, j, k)])
                                    * dx2_inv
                                    + (2.0f64.mul_add(-p, pressure[index(i, j + 1, k)])
                                        + pressure[index(i, j - 1, k)])
                                        * dy2_inv
                                    + (2.0f64.mul_add(-p, pressure[index(i, j, k + 1)])
                                        + pressure[index(i, j, k - 1)])
                                        * dz2_inv;
                            }
                        }
                    },
                );
            }
            4 => {
                // [−1, 16, −30, 16, −1] / (12h²) — CENTER=−5/2, NEAR=4/3, FAR=−1/12
                const CENTER: f64 = -5.0 / 2.0;
                const NEAR: f64 = 4.0 / 3.0;
                const FAR: f64 = -1.0 / 12.0;

                for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                    laplacian,
                    slab_len,
                    |i, lap_i| {
                        if i < 2 || i >= nx - 2 {
                            return;
                        }
                        for j in 2..ny - 2 {
                            for k in 2..nz - 2 {
                                let p_c = pressure[index(i, j, k)];
                                let d2x = FAR.mul_add(
                                    pressure[index(i - 2, j, k)] + pressure[index(i + 2, j, k)],
                                    NEAR.mul_add(
                                        pressure[index(i - 1, j, k)] + pressure[index(i + 1, j, k)],
                                        CENTER * p_c,
                                    ),
                                ) * dx2_inv;
                                let d2y = FAR.mul_add(
                                    pressure[index(i, j - 2, k)] + pressure[index(i, j + 2, k)],
                                    NEAR.mul_add(
                                        pressure[index(i, j - 1, k)] + pressure[index(i, j + 1, k)],
                                        CENTER * p_c,
                                    ),
                                ) * dy2_inv;
                                let d2z = FAR.mul_add(
                                    pressure[index(i, j, k - 2)] + pressure[index(i, j, k + 2)],
                                    NEAR.mul_add(
                                        pressure[index(i, j, k - 1)] + pressure[index(i, j, k + 1)],
                                        CENTER * p_c,
                                    ),
                                ) * dz2_inv;
                                lap_i[j * nz + k] = d2x + d2y + d2z;
                            }
                        }
                    },
                );
            }
            6 => {
                // [2,−27,270,−490,270,−27,2] / (180h²)
                const CENTER: f64 = -49.0 / 18.0;
                const NEAR: f64 = 3.0 / 2.0;
                const MID: f64 = -3.0 / 20.0;
                const FAR: f64 = 1.0 / 90.0;

                for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                    laplacian,
                    slab_len,
                    |i, lap_i| {
                        if i < 3 || i >= nx - 3 {
                            return;
                        }
                        for j in 3..ny - 3 {
                            for k in 3..nz - 3 {
                                let p_c = pressure[index(i, j, k)];
                                let d2x = FAR.mul_add(
                                    pressure[index(i - 3, j, k)] + pressure[index(i + 3, j, k)],
                                    MID.mul_add(
                                        pressure[index(i - 2, j, k)] + pressure[index(i + 2, j, k)],
                                        NEAR.mul_add(
                                            pressure[index(i - 1, j, k)]
                                                + pressure[index(i + 1, j, k)],
                                            CENTER * p_c,
                                        ),
                                    ),
                                ) * dx2_inv;
                                let d2y = FAR.mul_add(
                                    pressure[index(i, j - 3, k)] + pressure[index(i, j + 3, k)],
                                    MID.mul_add(
                                        pressure[index(i, j - 2, k)] + pressure[index(i, j + 2, k)],
                                        NEAR.mul_add(
                                            pressure[index(i, j - 1, k)]
                                                + pressure[index(i, j + 1, k)],
                                            CENTER * p_c,
                                        ),
                                    ),
                                ) * dy2_inv;
                                let d2z = FAR.mul_add(
                                    pressure[index(i, j, k - 3)] + pressure[index(i, j, k + 3)],
                                    MID.mul_add(
                                        pressure[index(i, j, k - 2)] + pressure[index(i, j, k + 2)],
                                        NEAR.mul_add(
                                            pressure[index(i, j, k - 1)]
                                                + pressure[index(i, j, k + 1)],
                                            CENTER * p_c,
                                        ),
                                    ),
                                ) * dz2_inv;
                                lap_i[j * nz + k] = d2x + d2y + d2z;
                            }
                        }
                    },
                );
            }
            _ => {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "spatial_order".to_owned(),
                    value: self.config.spatial_order as f64,
                    reason: "Westervelt FDTD supports only 2, 4, or 6".to_owned(),
                }));
            }
        }

        Ok(())
    }
}
