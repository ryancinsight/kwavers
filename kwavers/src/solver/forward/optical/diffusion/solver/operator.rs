//! 7-point finite-difference diffusion operator
//! `A Φ = ∇·(D∇Φ) − μₐΦ` with extrapolated-boundary handling per face.

use ndarray::Array3;

use super::DiffusionSolver;

impl DiffusionSolver {
    /// Apply diffusion operator: `A Φ = ∇·(D∇Φ) − μₐΦ`.
    ///
    /// Uses second-order finite differences with extrapolated boundary
    /// conditions. The face-averaged diffusion coefficient
    /// `D_{i±1/2} = ½(Dᵢ + D_{i±1})` ensures the discrete operator stays
    /// symmetric for heterogeneous media.
    pub(super) fn apply_operator(&self, fluence: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut result = Array3::zeros((nx, ny, nz));

        let bc = self.boundary_conditions();
        let dx2_inv = 1.0 / (self.grid.dx * self.grid.dx);
        let dy2_inv = 1.0 / (self.grid.dy * self.grid.dy);
        let dz2_inv = 1.0 / (self.grid.dz * self.grid.dz);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let phi_center = fluence[[i, j, k]];
                    let d_center = self.diffusion_coefficient[[i, j, k]];
                    let mu_a = self.absorption_coefficient[[i, j, k]];

                    let mut laplacian = 0.0;

                    // X-direction
                    if nx >= 2 {
                        if i == 0 {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.x_min, d_center, self.grid.dx);
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i + 1, j, k]]);
                            let phi_plus = fluence[[i + 1, j, k]];
                            let phi_minus = ghost_coeff * phi_center;
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_center * (phi_center - phi_minus))
                                * dx2_inv;
                        } else if i + 1 == nx {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.x_max, d_center, self.grid.dx);
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i - 1, j, k]]);
                            let phi_minus = fluence[[i - 1, j, k]];
                            let phi_plus = ghost_coeff * phi_center;
                            laplacian += d_center.mul_add(phi_plus - phi_center, -(d_minus * (phi_center - phi_minus)))
                                * dx2_inv;
                        } else {
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i - 1, j, k]]);
                            let phi_minus = fluence[[i - 1, j, k]];
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i + 1, j, k]]);
                            let phi_plus = fluence[[i + 1, j, k]];
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dx2_inv;
                        }
                    }

                    // Y-direction
                    if ny >= 2 {
                        if j == 0 {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.y_min, d_center, self.grid.dy);
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j + 1, k]]);
                            let phi_plus = fluence[[i, j + 1, k]];
                            let phi_minus = ghost_coeff * phi_center;
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_center * (phi_center - phi_minus))
                                * dy2_inv;
                        } else if j + 1 == ny {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.y_max, d_center, self.grid.dy);
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j - 1, k]]);
                            let phi_minus = fluence[[i, j - 1, k]];
                            let phi_plus = ghost_coeff * phi_center;
                            laplacian += d_center.mul_add(phi_plus - phi_center, -(d_minus * (phi_center - phi_minus)))
                                * dy2_inv;
                        } else {
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j - 1, k]]);
                            let phi_minus = fluence[[i, j - 1, k]];
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j + 1, k]]);
                            let phi_plus = fluence[[i, j + 1, k]];
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dy2_inv;
                        }
                    }

                    // Z-direction
                    if nz >= 2 {
                        if k == 0 {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.z_min, d_center, self.grid.dz);
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k + 1]]);
                            let phi_plus = fluence[[i, j, k + 1]];
                            let phi_minus = ghost_coeff * phi_center;
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_center * (phi_center - phi_minus))
                                * dz2_inv;
                        } else if k + 1 == nz {
                            let ghost_coeff =
                                Self::ghost_coefficient(bc.z_max, d_center, self.grid.dz);
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k - 1]]);
                            let phi_minus = fluence[[i, j, k - 1]];
                            let phi_plus = ghost_coeff * phi_center;
                            laplacian += d_center.mul_add(phi_plus - phi_center, -(d_minus * (phi_center - phi_minus)))
                                * dz2_inv;
                        } else {
                            let d_minus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k - 1]]);
                            let phi_minus = fluence[[i, j, k - 1]];
                            let d_plus =
                                0.5 * (d_center + self.diffusion_coefficient[[i, j, k + 1]]);
                            let phi_plus = fluence[[i, j, k + 1]];
                            laplacian += (d_plus * (phi_plus - phi_center)
                                - d_minus * (phi_center - phi_minus))
                                * dz2_inv;
                        }
                    }

                    result[[i, j, k]] = mu_a.mul_add(phi_center, -laplacian);
                }
            }
        }

        result
    }
}
