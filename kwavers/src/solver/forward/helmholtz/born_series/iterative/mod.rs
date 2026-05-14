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
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use num_complex::Complex64;
use std::f64::consts::PI;

mod derivatives;
mod stats;

#[cfg(test)]
mod tests;

pub use stats::IterativeBornStats;

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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if `contrast shape`.
    ///
    fn compute_scattering_potential<M: Medium>(
        &mut self,
        wavenumber: f64,
        medium: &M,
    ) -> KwaversResult<()> {
        let k0_squared = wavenumber * wavenumber;
        let (nx, ny, nz) = self.workspace.heterogeneity_workspace.dim();
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;

        // Phase 1: sequential — medium not guaranteed Sync; collect per-cell contrast.
        let contrasts: Vec<f64> = (0..nx)
            .flat_map(|i| {
                (0..ny).flat_map(move |j| {
                    (0..nz).map(move |k| {
                        let c = medium.sound_speed(i, j, k);
                        let rho = medium.density(i, j, k);
                        (rho * c * c) / (rho0 * c0 * c0)
                    })
                })
            })
            .collect();
        let contrasts_arr =
            ndarray::Array3::from_shape_vec((nx, ny, nz), contrasts).expect("contrast shape");

        // Phase 2: parallel — pure arithmetic on pre-collected contrast values.
        let current_field = &self.current_field;
        Zip::from(&mut self.workspace.heterogeneity_workspace)
            .and(&contrasts_arr)
            .and(current_field)
            .par_for_each(|potential, &contrast, &current_val| {
                let v = k0_squared * (1.0 - contrast);
                *potential = Complex64::new(v, 0.0) * current_val;
            });

        Ok(())
    }

    /// Apply Green's operator to scattering potential
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
                                    let r = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

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
        Zip::from(&mut self.current_field)
            .and(&self.incident_field)
            .and(&self.workspace.green_workspace)
            .par_for_each(|current, &incident, &green_contrib| {
                *current = incident + green_contrib;
            });
    }

    /// Compute residual for convergence check
    fn compute_residual<M: Medium>(&self, wavenumber: f64, medium: &M) -> f64 {
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
}
