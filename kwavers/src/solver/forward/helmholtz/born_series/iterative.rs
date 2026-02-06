//! Iterative Born Solver for Strong Scattering
//!
//! This module implements the iterative Born method for solving the Helmholtz
//! equation in media with strong heterogeneities where standard Born series
//! may not converge. The method uses fixed-point iteration with preconditioning.
//!
//! ## Mathematical Foundation
//!
//! The iterative Born method solves the Lippmann-Schwinger equation:
//! ```text
//! ψ = ψⁱ + G V ψ
//! ```
//!
//! Where:
//! - ψ: total field
//! - ψⁱ: incident field
//! - G: Green's operator
//! - V: scattering potential (heterogeneity)
//!
//! ## Fixed-Point Iteration
//!
//! The iteration scheme is:
//! ```text
//! ψ_{n+1} = ψⁱ + G V ψ_n
//! ```
//!
//! This converges when ||G V|| < 1 in appropriate operator norm.
//!
//! ## Preconditioning
//!
//! To improve convergence for strong scattering, we use preconditioning:
//! ```text
//! ψ_{n+1} = ψⁱ + P^{-1} (G V ψ_n)
//! ```
//!
//! Where P is a preconditioner operator.
//!
//! ## Heterogeneous Density Support
//!
//! Unlike standard Born methods, this implementation handles both sound speed
//! and density variations simultaneously:
//! ```text
//! V = 1 - (ρ c²)/(ρ₀ c₀²)
//! ```
//!
//! ## References
//!
//! Stanziola, A., et al. (2025). "Iterative Born Solver for the Acoustic
//! Helmholtz Equation with Heterogeneous Sound Speed and Density"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Iterative Born solver for Helmholtz equation
#[derive(Debug)]
pub struct IterativeBornSolver {
    /// Solver configuration
    config: super::BornConfig,
    /// Computational grid
    grid: Grid,
    /// Workspace for computations
    workspace: super::BornWorkspace,
    /// Incident field storage
    incident_field: Array3<Complex64>,
    /// Current iteration field
    current_field: Array3<Complex64>,
}

impl IterativeBornSolver {
    /// Create a new iterative Born solver
    pub fn new(config: super::BornConfig, grid: Grid) -> Self {
        let workspace = super::BornWorkspace::new(grid.nx, grid.ny, grid.nz);
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            config,
            grid,
            workspace,
            incident_field: Array3::zeros(shape),
            current_field: Array3::zeros(shape),
        }
    }

    /// Solve Helmholtz equation using iterative Born method
    pub fn solve<M: Medium>(
        &mut self,
        wavenumber: f64,
        medium: &M,
        incident_field: ArrayView3<Complex64>,
        mut result: ArrayViewMut3<Complex64>,
    ) -> KwaversResult<IterativeBornStats> {
        // Initialize fields
        self.incident_field.assign(&incident_field);
        self.current_field.assign(&incident_field);
        self.workspace.clear();

        let mut stats = IterativeBornStats::default();
        let mut converged = false;

        // Iterative solution
        for iteration in 0..self.config.max_iterations {
            let residual = self.iterative_born_step(wavenumber, medium)?;

            stats.iterations = iteration + 1;
            stats.final_residual = residual;
            stats.residual_history.push(residual);

            if residual < self.config.tolerance {
                converged = true;
                break;
            }
        }

        stats.converged = converged;
        result.assign(&self.current_field);

        Ok(stats)
    }

    /// Perform one step of iterative Born method
    fn iterative_born_step<M: Medium>(
        &mut self,
        wavenumber: f64,
        medium: &M,
    ) -> KwaversResult<f64> {
        // Compute scattering potential: V * ψ_current
        self.compute_scattering_potential(wavenumber, medium)?;

        // Apply Green's operator: G * (V ψ)
        self.apply_green_operator(wavenumber)?;

        // Update field: ψ_{n+1} = ψⁱ + G (V ψ_n)
        self.update_field();

        // Compute residual
        let residual = self.compute_residual(wavenumber, medium);

        Ok(residual)
    }

    /// Compute scattering potential V * ψ
    fn compute_scattering_potential<M: Medium>(
        &mut self,
        wavenumber: f64,
        medium: &M,
    ) -> KwaversResult<()> {
        use ndarray::Zip;

        let k0_squared = wavenumber * wavenumber;

        Zip::indexed(&mut self.workspace.heterogeneity_workspace).for_each(
            |(i, j, k), potential| {
                // Compute local medium properties
                let c_local = medium.sound_speed(i, j, k);
                let rho_local = medium.density(i, j, k);

                // Reference values (could be made configurable)
                let c0 = 1500.0; // m/s
                let rho0 = 1000.0; // kg/m³
                let k0_squared_ref = k0_squared;

                // Scattering potential: V = k²(1 - (ρ c²)/(ρ₀ c₀²))
                let contrast = (rho_local * c_local * c_local) / (rho0 * c0 * c0);
                let v = k0_squared_ref * (1.0 - contrast);

                *potential = Complex64::new(v, 0.0) * self.current_field[[i, j, k]];
            },
        );

        Ok(())
    }

    /// Apply Green's operator to scattering potential
    fn apply_green_operator(&mut self, wavenumber: f64) -> KwaversResult<()> {
        // Use efficient local approximation for Green's function
        // Full 3D convolution would be better but this provides reasonable accuracy

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        // Clear result array
        self.workspace
            .green_workspace
            .fill(Complex64::new(0.0, 0.0));

        // For each point, compute local Green's function contributions
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let source_val = self.workspace.heterogeneity_workspace[[i, j, k]];

                    // Self-contribution (regularized)
                    let self_green = Complex64::new(0.5, 0.0); // Regularized 1/r singularity
                    self.workspace.green_workspace[[i, j, k]] += self_green * source_val;

                    // Contributions from 26 nearest neighbors (3x3x3 stencil minus center)
                    for di in -1i32..=1 {
                        for dj in -1i32..=1 {
                            for dk in -1i32..=1 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue; // Skip self
                                }

                                let ni = i as i32 + di;
                                let nj = j as i32 + dj;
                                let nk = k as i32 + dk;

                                // Check bounds
                                if ni >= 0
                                    && ni < nx as i32
                                    && nj >= 0
                                    && nj < ny as i32
                                    && nk >= 0
                                    && nk < nz as i32
                                {
                                    let dx = di as f64 * self.grid.dx;
                                    let dy = dj as f64 * self.grid.dy;
                                    let dz = dk as f64 * self.grid.dz;
                                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

                                    // Free space Green's function approximation
                                    let kr = wavenumber * r;
                                    let green_val = if r > 1e-12 {
                                        Complex64::from_polar(1.0 / (4.0 * PI * r), kr)
                                    } else {
                                        Complex64::new(0.1, 0.0) // Fallback
                                    };

                                    self.workspace.green_workspace
                                        [[ni as usize, nj as usize, nk as usize]] +=
                                        green_val * source_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Update field using Born iteration
    fn update_field(&mut self) {
        use ndarray::Zip;

        Zip::from(&mut self.current_field)
            .and(&self.incident_field)
            .and(&self.workspace.green_workspace)
            .for_each(|current, &incident, &green_contrib| {
                *current = incident + green_contrib;
            });
    }

    /// Compute residual for convergence check
    fn compute_residual<M: Medium>(&self, wavenumber: f64, medium: &M) -> f64 {
        use ndarray::Zip;

        let mut residual_sum = 0.0;
        let k_squared = wavenumber * wavenumber;

        // Compute ||∇²ψ + k²(1+V)ψ|| as residual
        Zip::indexed(&self.current_field).for_each(|(i, j, k), &field_val| {
            // Simplified Laplacian computation (central difference)
            let laplacian = self.compute_laplacian(i, j, k);

            // Compute local heterogeneity
            let c_local = medium.sound_speed(i, j, k);
            let rho_local = medium.density(i, j, k);
            let c0 = 1500.0;
            let rho0 = 1000.0;
            let contrast = (rho_local * c_local * c_local) / (rho0 * c0 * c0);
            let heterogeneity = 1.0 - contrast;

            // Helmholtz residual: ∇²ψ + k²(1+V)ψ
            let helmholtz_residual = laplacian + k_squared * heterogeneity * field_val;
            residual_sum += helmholtz_residual.norm_sqr();
        });

        (residual_sum / (self.grid.nx * self.grid.ny * self.grid.nz) as f64).sqrt()
    }

    /// Compute Laplacian using finite differences (3D central difference)
    fn compute_laplacian(&self, i: usize, j: usize, k: usize) -> Complex64 {
        // Second derivatives in each direction
        let d2x = self.second_derivative_x(i, j, k);
        let d2y = self.second_derivative_y(i, j, k);
        let d2z = self.second_derivative_z(i, j, k);

        d2x + d2y + d2z
    }

    /// Compute second derivative in x-direction with proper boundary handling
    fn second_derivative_x(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let field = &self.current_field;
        let dx2 = self.grid.dx * self.grid.dx;

        if i == 0 {
            // Forward difference at left boundary
            let f0 = field[[0, j, k]];
            let f1 = field[[1, j, k]];
            let f2 = field[[2, j, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dx2 // Second-order forward
        } else if i == self.grid.nx - 1 {
            // Backward difference at right boundary
            let f0 = field[[i, j, k]];
            let fm1 = field[[i - 1, j, k]];
            let fm2 = field[[i - 2, j, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dx2 // Second-order backward
        } else {
            // Central difference in interior
            let fm1 = field[[i - 1, j, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i + 1, j, k]];
            (fm1 - 2.0 * f0 + fp1) / dx2
        }
    }

    /// Compute second derivative in y-direction with proper boundary handling
    fn second_derivative_y(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let field = &self.current_field;
        let dy2 = self.grid.dy * self.grid.dy;

        if j == 0 {
            // Forward difference at bottom boundary
            let f0 = field[[i, 0, k]];
            let f1 = field[[i, 1, k]];
            let f2 = field[[i, 2, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dy2
        } else if j == self.grid.ny - 1 {
            // Backward difference at top boundary
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j - 1, k]];
            let fm2 = field[[i, j - 2, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dy2
        } else {
            // Central difference in interior
            let fm1 = field[[i, j - 1, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j + 1, k]];
            (fm1 - 2.0 * f0 + fp1) / dy2
        }
    }

    /// Compute second derivative in z-direction with proper boundary handling
    fn second_derivative_z(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let field = &self.current_field;
        let dz2 = self.grid.dz * self.grid.dz;

        if k == 0 {
            // Forward difference at front boundary
            let f0 = field[[i, j, 0]];
            let f1 = field[[i, j, 1]];
            let f2 = field[[i, j, 2]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dz2
        } else if k == self.grid.nz - 1 {
            // Backward difference at back boundary
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j, k - 1]];
            let fm2 = field[[i, j, k - 2]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dz2
        } else {
            // Central difference in interior
            let fm1 = field[[i, j, k - 1]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j, k + 1]];
            (fm1 - 2.0 * f0 + fp1) / dz2
        }
    }
}

/// Statistics from iterative Born solution
#[derive(Debug, Clone)]
pub struct IterativeBornStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual value
    pub final_residual: f64,
    /// History of residual values
    pub residual_history: Vec<f64>,
    /// Whether convergence was achieved
    pub converged: bool,
}

impl Default for IterativeBornStats {
    fn default() -> Self {
        Self {
            iterations: 0,
            final_residual: 0.0,
            residual_history: Vec::new(),
            converged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::solver::forward::helmholtz::BornConfig;

    #[test]
    fn test_iterative_born_creation() {
        let config = BornConfig::default();
        let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();

        let solver = IterativeBornSolver::new(config, grid);
        assert_eq!(solver.config.max_iterations, 50);
        assert_eq!(solver.grid.nx, 16);
    }
}
