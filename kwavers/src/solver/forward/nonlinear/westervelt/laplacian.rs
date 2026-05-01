//! Finite-difference Laplacian for the Westervelt FDTD solver.
//!
//! Operates in-place on the solver's pre-allocated `laplacian` workspace.
//! Supports 2nd- and 4th-order central-difference stencils; an unsupported
//! order silently downgrades to second-order.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;

use super::WesterveltFdtd;

impl WesterveltFdtd {
    /// Calculate the Laplacian using finite differences
    pub(super) fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
        let pressure = &self.pressure;
        let laplacian = &mut self.laplacian;

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);
        let dx2_inv = 1.0 / (dx * dx);
        let dy2_inv = 1.0 / (dy * dy);
        let dz2_inv = 1.0 / (dz * dz);

        match self.config.spatial_order {
            2 => {
                // Second-order accurate using safe indexing
                for i in 1..nx - 1 {
                    for j in 1..ny - 1 {
                        for k in 1..nz - 1 {
                            let p = pressure[(i, j, k)];
                            let px_m = pressure[(i - 1, j, k)];
                            let px_p = pressure[(i + 1, j, k)];
                            let py_m = pressure[(i, j - 1, k)];
                            let py_p = pressure[(i, j + 1, k)];
                            let pz_m = pressure[(i, j, k - 1)];
                            let pz_p = pressure[(i, j, k + 1)];

                            laplacian[(i, j, k)] = (px_p - 2.0 * p + px_m) * dx2_inv
                                + (py_p - 2.0 * p + py_m) * dy2_inv
                                + (pz_p - 2.0 * p + pz_m) * dz2_inv;
                        }
                    }
                }
            }
            4 => {
                // Fourth-order accurate stencil coefficients from constants
                const FD4_COEFF_0: f64 = -5.0 / 2.0;
                const FD4_COEFF_1: f64 = 4.0 / 3.0;
                const FD4_COEFF_2: f64 = -1.0 / 12.0;
                const C0: f64 = FD4_COEFF_0;
                const C1: f64 = FD4_COEFF_1;
                const C2: f64 = FD4_COEFF_2;

                for i in 2..nx - 2 {
                    for j in 2..ny - 2 {
                        for k in 2..nz - 2 {
                            let p_c = pressure[(i, j, k)];

                            // X-direction stencil
                            let p_xm2 = pressure[(i - 2, j, k)];
                            let p_xm1 = pressure[(i - 1, j, k)];
                            let p_xp1 = pressure[(i + 1, j, k)];
                            let p_xp2 = pressure[(i + 2, j, k)];

                            let d2_dx2 =
                                (C0 * p_xm2 + C1 * p_xm1 + C2 * p_c + C1 * p_xp1 + C0 * p_xp2)
                                    * dx2_inv;

                            // Y-direction stencil
                            let p_ym2 = pressure[(i, j - 2, k)];
                            let p_ym1 = pressure[(i, j - 1, k)];
                            let p_yp1 = pressure[(i, j + 1, k)];
                            let p_yp2 = pressure[(i, j + 2, k)];

                            let d2_dy2 =
                                (C0 * p_ym2 + C1 * p_ym1 + C2 * p_c + C1 * p_yp1 + C0 * p_yp2)
                                    * dy2_inv;

                            // Z-direction stencil
                            let p_zm2 = pressure[(i, j, k - 2)];
                            let p_zm1 = pressure[(i, j, k - 1)];
                            let p_zp1 = pressure[(i, j, k + 1)];
                            let p_zp2 = pressure[(i, j, k + 2)];

                            let d2_dz2 =
                                (C0 * p_zm2 + C1 * p_zm1 + C2 * p_c + C1 * p_zp1 + C0 * p_zp2)
                                    * dz2_inv;

                            laplacian[(i, j, k)] = d2_dx2 + d2_dy2 + d2_dz2;
                        }
                    }
                }
            }
            _ => {
                // Default to second-order for unsupported orders
                self.config.spatial_order = 2;
                self.calculate_laplacian(grid)?;
            }
        }

        Ok(())
    }
}
