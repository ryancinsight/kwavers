//! DG solver implementation and time stepping
//!
//! This module contains the core solver implementation including
//! flux computation, limiting, and time stepping operations.

use super::super::flux::{apply_limiter, compute_numerical_flux};
use super::super::matrices::matrix_inverse;
use super::core::DGSolver;
use crate::error::KwaversError;
use crate::KwaversResult;
use ndarray::Array3;

impl DGSolver {
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

        // Compute volume integrals
        self.compute_volume_integrals(&coeffs_copy, &mut rhs, &mass_inv, wave_speed)?;

        // Compute surface integrals
        self.compute_surface_integrals(&coeffs_copy, &mut rhs, &mass_inv, wave_speed)?;

        // Apply limiter if enabled
        self.apply_limiter(&mut rhs)?;

        // Update solution: u^{n+1} = u^n + dt * RHS
        let coeffs = self.modal_coefficients.as_mut().unwrap();
        for i in 0..coeffs.len() {
            coeffs.as_slice_mut().unwrap()[i] += dt * rhs.as_slice().unwrap()[i];
        }

        // Project back to grid
        self.project_to_grid(field)?;

        Ok(())
    }

    /// Compute volume integrals for DG formulation
    fn compute_volume_integrals(
        &self,
        coeffs: &Array3<f64>,
        rhs: &mut Array3<f64>,
        mass_inv: &ndarray::Array2<f64>,
        wave_speed: f64,
    ) -> KwaversResult<()> {
        let n_elements = coeffs.shape()[0];

        for elem in 0..n_elements {
            // Compute flux within element
            for node in 0..self.n_nodes {
                let u = coeffs[(elem, node, 0)];
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

        Ok(())
    }

    /// Compute surface integrals for DG formulation
    fn compute_surface_integrals(
        &self,
        coeffs: &Array3<f64>,
        rhs: &mut Array3<f64>,
        _mass_inv: &ndarray::Array2<f64>,
        wave_speed: f64,
    ) -> KwaversResult<()> {
        let n_elements = coeffs.shape()[0];

        // Surface integral: M^{-1} * L * (f* - f)
        for elem in 0..n_elements - 1 {
            // Get states at interface
            let left_state = coeffs[(elem, self.n_nodes - 1, 0)];
            let right_state = coeffs[(elem + 1, 0, 0)];

            // Compute numerical flux
            let flux_star = self.compute_numerical_flux(left_state, right_state, wave_speed)?;

            // Add surface contribution
            let left_flux = wave_speed * left_state;
            let right_flux = wave_speed * right_state;

            // Left element contribution (right boundary)
            for i in 0..self.n_nodes {
                rhs[(elem, i, 0)] += self.lift_matrix[(i, 1)] * (flux_star - left_flux);
            }

            // Right element contribution (left boundary)
            for i in 0..self.n_nodes {
                rhs[(elem + 1, i, 0)] += self.lift_matrix[(i, 0)] * (flux_star - right_flux);
            }
        }

        Ok(())
    }
}
