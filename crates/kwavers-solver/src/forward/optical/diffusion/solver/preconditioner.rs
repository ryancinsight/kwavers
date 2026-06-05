//! Jacobi preconditioner — the inverse diagonal of the diffusion operator
//! `A = ∇·(D∇·) − μₐ·`. Used by the conjugate-gradient driver in `solve.rs`
//! to accelerate convergence on heterogeneous media.

use ndarray::Array3;

use super::DiffusionSolver;

impl DiffusionSolver {
    /// Compute Jacobi preconditioner (inverse of system-matrix diagonal).
    pub(super) fn compute_preconditioner(&self) -> Array3<f64> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut preconditioner = Array3::zeros((nx, ny, nz));

        let bc = self.boundary_conditions();
        let dx2_inv = 1.0 / (self.grid.dx * self.grid.dx);
        let dy2_inv = 1.0 / (self.grid.dy * self.grid.dy);
        let dz2_inv = 1.0 / (self.grid.dz * self.grid.dz);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let d = self.diffusion_coefficient[[i, j, k]];
                    let mu_a = self.absorption_coefficient[[i, j, k]];

                    let ghost_coeff_x_min = Self::ghost_coefficient(bc.x_min, d, self.grid.dx);
                    let ghost_coeff_x_max = Self::ghost_coefficient(bc.x_max, d, self.grid.dx);
                    let ghost_coeff_y_min = Self::ghost_coefficient(bc.y_min, d, self.grid.dy);
                    let ghost_coeff_y_max = Self::ghost_coefficient(bc.y_max, d, self.grid.dy);
                    let ghost_coeff_z_min = Self::ghost_coefficient(bc.z_min, d, self.grid.dz);
                    let ghost_coeff_z_max = Self::ghost_coefficient(bc.z_max, d, self.grid.dz);

                    let d_x_minus = if i > 0 {
                        0.5 * (d + self.diffusion_coefficient[[i - 1, j, k]])
                    } else {
                        d * (1.0 - ghost_coeff_x_min)
                    };
                    let d_x_plus = if i + 1 < nx {
                        0.5 * (d + self.diffusion_coefficient[[i + 1, j, k]])
                    } else {
                        d * (1.0 - ghost_coeff_x_max)
                    };
                    let d_y_minus = if j > 0 {
                        0.5 * (d + self.diffusion_coefficient[[i, j - 1, k]])
                    } else {
                        d * (1.0 - ghost_coeff_y_min)
                    };
                    let d_y_plus = if j + 1 < ny {
                        0.5 * (d + self.diffusion_coefficient[[i, j + 1, k]])
                    } else {
                        d * (1.0 - ghost_coeff_y_max)
                    };
                    let d_z_minus = if k > 0 {
                        0.5 * (d + self.diffusion_coefficient[[i, j, k - 1]])
                    } else {
                        d * (1.0 - ghost_coeff_z_min)
                    };
                    let d_z_plus = if k + 1 < nz {
                        0.5 * (d + self.diffusion_coefficient[[i, j, k + 1]])
                    } else {
                        d * (1.0 - ghost_coeff_z_max)
                    };

                    let diagonal = (d_z_minus + d_z_plus).mul_add(
                        dz2_inv,
                        (d_x_minus + d_x_plus).mul_add(dx2_inv, (d_y_minus + d_y_plus) * dy2_inv),
                    ) + mu_a;

                    preconditioner[[i, j, k]] = if diagonal > 1e-30 {
                        1.0 / diagonal
                    } else {
                        1.0
                    };
                }
            }
        }

        preconditioner
    }
}
