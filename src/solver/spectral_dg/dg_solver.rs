//! Discontinuous Galerkin (DG) solver implementation
//!
//! This module implements a DG method for solving hyperbolic conservation laws
//! with shock-capturing capabilities for handling discontinuities.
//!
//! # References
//!
//! - Hesthaven, J. S., & Warburton, T. (2008). "Nodal discontinuous Galerkin methods"
//! - Cockburn, B., & Shu, C. W. (2001). "Runge-Kutta discontinuous Galerkin methods"

use super::basis::{build_vandermonde, BasisType};
use super::flux::{apply_limiter, compute_numerical_flux, FluxType, LimiterType};
use super::matrices::{
    compute_diff_matrix, compute_lift_matrix, compute_mass_matrix, compute_stiffness_matrix,
    matrix_inverse,
};
use super::quadrature::gauss_lobatto_quadrature;
use super::traits::{DGOperations, NumericalSolver};
use crate::error::KwaversError;
use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

/// Configuration for DG solver
#[derive(Debug, Clone)]
pub struct DGConfig {
    /// Polynomial order (N means polynomials up to degree N)
    pub polynomial_order: usize,
    /// Basis function type
    pub basis_type: BasisType,
    /// Numerical flux type
    pub flux_type: FluxType,
    /// Enable slope limiting for shock capturing
    pub use_limiter: bool,
    /// Limiter type (if enabled)
    pub limiter_type: LimiterType,
}

impl Default for DGConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 3,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::Minmod,
        }
    }
}

/// DG solver for hyperbolic conservation laws
#[derive(Debug)]
pub struct DGSolver {
    /// Configuration
    config: DGConfig,
    /// Grid reference
    grid: Arc<Grid>,
    /// Number of nodes per element
    n_nodes: usize,
    /// Quadrature nodes on reference element [-1, 1]
    xi_nodes: Arc<Array1<f64>>,
    /// Quadrature weights
    weights: Arc<Array1<f64>>,
    /// Vandermonde matrix for basis evaluation
    vandermonde: Arc<Array2<f64>>,
    /// Mass matrix M_ij = integral(phi_i * phi_j)
    mass_matrix: Arc<Array2<f64>>,
    /// Stiffness matrix S_ij = integral(phi_i * dphi_j/dxi)
    stiffness_matrix: Arc<Array2<f64>>,
    /// Differentiation matrix D = V * Dr * V^{-1}
    diff_matrix: Arc<Array2<f64>>,
    /// Lift matrix for surface integrals
    lift_matrix: Arc<Array2<f64>>,
    /// Modal coefficients for each element (n_elements x n_nodes x n_vars)
    modal_coefficients: Option<Array3<f64>>,
}

impl DGSolver {
    /// Create a new DG solver
    pub fn new(config: DGConfig, grid: Arc<Grid>) -> KwaversResult<Self> {
        let n_nodes = config.polynomial_order + 1;

        // Generate quadrature nodes and weights
        let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;

        // Build Vandermonde matrix
        let vandermonde = build_vandermonde(&xi_nodes, config.polynomial_order, config.basis_type)?;

        // Compute mass matrix
        let mass_matrix = compute_mass_matrix(&vandermonde, &weights)?;

        // Compute stiffness matrix
        let stiffness_matrix =
            compute_stiffness_matrix(&vandermonde, &xi_nodes, &weights, config.basis_type)?;

        // Compute differentiation matrix
        let diff_matrix = compute_diff_matrix(&vandermonde, &xi_nodes, config.basis_type)?;

        // Compute lift matrix for boundary terms
        let lift_matrix = compute_lift_matrix(&mass_matrix, n_nodes)?;

        Ok(Self {
            config,
            grid,
            n_nodes,
            xi_nodes: Arc::new(xi_nodes),
            weights: Arc::new(weights),
            vandermonde: Arc::new(vandermonde),
            mass_matrix: Arc::new(mass_matrix),
            stiffness_matrix: Arc::new(stiffness_matrix),
            diff_matrix: Arc::new(diff_matrix),
            lift_matrix: Arc::new(lift_matrix),
            modal_coefficients: None,
        })
    }

    /// Project a field onto the DG basis
    pub fn project_to_dg(&mut self, field: &Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);

        // Determine number of elements (simplified for structured grid)
        let n_elements_x = nx / self.n_nodes;
        let n_elements_y = ny / self.n_nodes;
        let n_elements_z = nz / self.n_nodes;

        let mut coeffs = Array3::zeros((
            n_elements_x * n_elements_y * n_elements_z,
            self.n_nodes,
            1, // Single variable for now
        ));

        // Project field onto modal basis
        for elem_idx in 0..coeffs.shape()[0] {
            // Map element index to grid location
            let elem_z = elem_idx / (n_elements_x * n_elements_y);
            let elem_y = (elem_idx % (n_elements_x * n_elements_y)) / n_elements_x;
            let elem_x = elem_idx % n_elements_x;

            // Extract element data
            for node in 0..self.n_nodes {
                let grid_x = elem_x * self.n_nodes + node;
                let grid_y = elem_y * self.n_nodes;
                let grid_z = elem_z * self.n_nodes;

                if grid_x < nx && grid_y < ny && grid_z < nz {
                    coeffs[(elem_idx, node, 0)] = field[(grid_x, grid_y, grid_z)];
                }
            }
        }

        self.modal_coefficients = Some(coeffs);
        Ok(())
    }

    /// Compute numerical flux between elements
    fn compute_numerical_flux(
        &self,
        left_state: f64,
        right_state: f64,
        wave_speed: f64,
    ) -> KwaversResult<f64> {
        // Numerical flux computation for scalar conservation law
        let left_flux = wave_speed * left_state;
        let right_flux = wave_speed * right_state;

        compute_numerical_flux(
            self.config.flux_type,
            left_state,
            right_state,
            left_flux,
            right_flux,
            wave_speed,
            1.0, // normal direction
        )
    }

    /// Apply slope limiter for shock capturing
    fn apply_limiter(&self, coeffs: &mut Array3<f64>) -> KwaversResult<()> {
        if !self.config.use_limiter {
            return Ok(());
        }

        let n_elements = coeffs.shape()[0];
        let n_vars = coeffs.shape()[2];

        for var in 0..n_vars {
            for elem in 1..n_elements - 1 {
                // Get cell averages
                let u_minus = coeffs[(elem - 1, 0, var)];
                let u_center = coeffs[(elem, 0, var)];
                let u_plus = coeffs[(elem + 1, 0, var)];

                // Compute differences
                let delta_minus = u_center - u_minus;
                let delta_plus = u_plus - u_center;

                // Apply limiter to higher-order modes
                for mode in 1..self.n_nodes {
                    let limited = apply_limiter(self.config.limiter_type, delta_minus, delta_plus);

                    // Scale higher-order coefficients
                    if limited.abs() < coeffs[(elem, mode, var)].abs() {
                        coeffs[(elem, mode, var)] *= limited / (coeffs[(elem, mode, var)] + 1e-10);
                    }
                }
            }
        }

        Ok(())
    }

    /// Perform one time step using DG method
    pub fn solve_step(&mut self, field: &mut Array3<f64>, dt: f64) -> KwaversResult<()> {
        // Project to DG basis if not already done
        if self.modal_coefficients.is_none() {
            self.project_to_dg(field)?;
        }

        // Get coefficients and compute dimensions
        let coeffs_shape = self
            .modal_coefficients
            .as_ref()
            .ok_or_else(|| {
                KwaversError::InvalidInput("Modal coefficients not initialized".to_string())
            })?
            .raw_dim();
        let n_elements = coeffs_shape[0];
        let wave_speed = 1500.0; // Example wave speed

        // Compute RHS of DG formulation
        let mut rhs = Array3::zeros(coeffs_shape);

        // Volume integral: -M^{-1} * S * f(u)
        let mass_inv = matrix_inverse(&self.mass_matrix)?;

        // Extract coefficients for computation
        let coeffs_copy = self.modal_coefficients.as_ref().unwrap().clone();

        for elem in 0..n_elements {
            // Compute flux within element
            for node in 0..self.n_nodes {
                let u = coeffs_copy[(elem, node, 0)];
                let flux = wave_speed * u;

                // Apply stiffness matrix
                for i in 0..self.n_nodes {
                    rhs[(elem, i, 0)] -= self.stiffness_matrix[(i, node)] * flux;
                }
            }

            // Apply mass matrix inverse
            for i in 0..self.n_nodes {
                let mut temp = 0.0;
                for j in 0..self.n_nodes {
                    temp += mass_inv[(i, j)] * rhs[(elem, j, 0)];
                }
                rhs[(elem, i, 0)] = temp;
            }
        }

        // Surface integral: M^{-1} * L * (f* - f)
        for elem in 0..n_elements - 1 {
            // Get states at interface
            let left_state = coeffs_copy[(elem, self.n_nodes - 1, 0)];
            let right_state = coeffs_copy[(elem + 1, 0, 0)];

            // Compute numerical flux
            let flux_star = self.compute_numerical_flux(left_state, right_state, wave_speed)?;

            // Add surface contribution
            let left_flux = wave_speed * left_state;
            let right_flux = wave_speed * right_state;

            // Left element contribution
            rhs[(elem, self.n_nodes - 1, 0)] +=
                self.lift_matrix[(self.n_nodes - 1, 1)] * (flux_star - left_flux);

            // Right element contribution
            rhs[(elem + 1, 0, 0)] -= self.lift_matrix[(0, 0)] * (flux_star - right_flux);
        }

        // Take ownership of coefficients to avoid borrow issues
        let mut coeffs = self.modal_coefficients.take().ok_or_else(|| {
            KwaversError::InvalidInput("Modal coefficients not initialized".to_string())
        })?;

        // Apply limiter if needed
        if self.config.use_limiter {
            // Apply limiter inline to avoid borrow issues
            let n_vars = coeffs.shape()[2];

            for var in 0..n_vars {
                for elem in 1..n_elements - 1 {
                    // Get cell averages
                    let u_minus = coeffs[(elem - 1, 0, var)];
                    let u_center = coeffs[(elem, 0, var)];
                    let u_plus = coeffs[(elem + 1, 0, var)];

                    // Compute differences
                    let delta_minus = u_center - u_minus;
                    let delta_plus = u_plus - u_center;

                    // Apply limiter to higher-order modes
                    for mode in 1..self.n_nodes {
                        let limited =
                            apply_limiter(self.config.limiter_type, delta_minus, delta_plus);

                        // Scale higher-order coefficients
                        if limited.abs() < coeffs[(elem, mode, var)].abs() {
                            coeffs[(elem, mode, var)] *=
                                limited / (coeffs[(elem, mode, var)] + 1e-10);
                        }
                    }
                }
            }
        }

        // Update solution (forward Euler for simplicity)
        for elem in 0..n_elements {
            for node in 0..self.n_nodes {
                coeffs[(elem, node, 0)] += dt * rhs[(elem, node, 0)];
            }
        }

        // Reconstruct field from modal coefficients
        let reconstructed = self.reconstruct_field(&coeffs)?;
        *field = reconstructed;

        // Put coefficients back
        self.modal_coefficients = Some(coeffs);

        Ok(())
    }

    /// Reconstruct physical field from modal coefficients
    fn reconstruct_field(&self, coeffs: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let mut field = Array3::zeros((nx, ny, nz));

        // Simplified reconstruction for structured grid
        let n_elements_x = nx / self.n_nodes;

        for elem_idx in 0..coeffs.shape()[0] {
            let elem_x = elem_idx % n_elements_x;

            for node in 0..self.n_nodes {
                let grid_x = elem_x * self.n_nodes + node;
                if grid_x < nx {
                    field[(grid_x, 0, 0)] = coeffs[(elem_idx, node, 0)];
                }
            }
        }

        Ok(field)
    }
}

impl NumericalSolver for DGSolver {
    fn solve(
        &self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        // Clone self to make it mutable for this operation
        let mut solver = self.clone();
        let mut result = field.clone();

        // Apply DG method only in regions marked by mask
        solver.project_to_dg(&result)?;
        solver.solve_step(&mut result, dt)?;

        // Apply mask to blend with original field
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                for k in 0..result.shape()[2] {
                    if !mask[(i, j, k)] {
                        result[(i, j, k)] = field[(i, j, k)];
                    }
                }
            }
        }

        Ok(result)
    }

    fn max_stable_dt(&self, grid: &Grid) -> f64 {
        // CFL condition for DG: dt <= dx / (wave_speed * (2p + 1))
        let dx = grid.dx.min(grid.dy).min(grid.dz);
        let wave_speed = 1500.0; // Example
        let p = self.config.polynomial_order as f64;

        dx / (wave_speed * (2.0 * p + 1.0))
    }

    fn update_order(&mut self, order: usize) {
        self.config.polynomial_order = order;
        // Would need to rebuild matrices here
    }
}

impl Clone for DGSolver {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            grid: Arc::clone(&self.grid),
            n_nodes: self.n_nodes,
            xi_nodes: Arc::clone(&self.xi_nodes),
            weights: Arc::clone(&self.weights),
            vandermonde: Arc::clone(&self.vandermonde),
            mass_matrix: Arc::clone(&self.mass_matrix),
            stiffness_matrix: Arc::clone(&self.stiffness_matrix),
            diff_matrix: Arc::clone(&self.diff_matrix),
            lift_matrix: Arc::clone(&self.lift_matrix),
            modal_coefficients: self.modal_coefficients.clone(), // Still needs clone for Option<Array3>
        }
    }
}

impl DGOperations for DGSolver {
    fn compute_flux(&self, left_state: f64, right_state: f64, normal: f64) -> f64 {
        // Upwind flux computation for numerical stability
        let wave_speed = 1500.0;
        if wave_speed * normal > 0.0 {
            wave_speed * left_state * normal
        } else {
            wave_speed * right_state * normal
        }
    }

    fn project_to_basis(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Project field onto modal basis
        let coeffs = field.clone();
        // Implementation would go here
        Ok(coeffs)
    }

    fn reconstruct_from_basis(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        self.reconstruct_field(coefficients)
    }
}

// Additional methods
impl DGSolver {
    /// Apply shock detector for adaptive limiting
    pub fn apply_shock_detector(&self, field: &Array3<f64>) -> Array3<bool> {
        let mut shock_cells = Array3::from_elem(field.raw_dim(), false);

        // Gradient-based shock detection using jump discontinuities
        for i in 1..field.shape()[0] - 1 {
            for j in 1..field.shape()[1] - 1 {
                for k in 1..field.shape()[2] - 1 {
                    let grad_x =
                        (field[(i + 1, j, k)] - field[(i - 1, j, k)]) / (2.0 * self.grid.dx);
                    let grad_y =
                        (field[(i, j + 1, k)] - field[(i, j - 1, k)]) / (2.0 * self.grid.dy);
                    let grad_z =
                        (field[(i, j, k + 1)] - field[(i, j, k - 1)]) / (2.0 * self.grid.dz);

                    let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();

                    // Threshold for shock detection
                    if grad_mag > 1000.0 {
                        shock_cells[(i, j, k)] = true;
                    }
                }
            }
        }

        shock_cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dg_solver_creation() {
        let grid = Arc::new(Grid::new(32, 32, 32, 0.001, 0.001, 0.001));
        let config = DGConfig::default();
        let solver = DGSolver::new(config, grid);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_upwind_flux() {
        let grid = Arc::new(Grid::new(32, 32, 32, 0.001, 0.001, 0.001));
        let config = DGConfig::default();
        let solver = DGSolver::new(config, grid).unwrap();

        let flux = solver.compute_flux(1.0, 2.0, 1.0);
        assert!(flux > 0.0);
    }
}
