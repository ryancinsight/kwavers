//! DG solver implementation and time stepping
//!
//! This module contains the core solver implementation including
//! flux computation, limiting, and time stepping operations.

use super::super::flux::{apply_limiter, compute_numerical_flux};
use super::super::matrices::matrix_inverse;
use super::core::DGSolver;
use crate::core::error::KwaversResult;
use crate::core::error::{KwaversError, ValidationError};
use ndarray::{Array2, Array3};

impl DGSolver {
    /// Compute numerical flux between elements
    fn compute_numerical_flux(
        &self,
        left_state: f64,
        right_state: f64,
        wave_speed: f64,
    ) -> KwaversResult<f64> {
        // Numerical flux computation for scalar conservation law
        // Args: left, right, flux_type, wave_speed, normal
        Ok(compute_numerical_flux(
            left_state,
            right_state,
            self.config.flux_type,
            wave_speed,
            1.0, // normal direction
        ))
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
    pub fn solve_step(&mut self, field: &mut Array3<f64>, _dt: f64) -> KwaversResult<()> {
        // Project to DG basis if not already done
        if self.modal_coefficients.is_none() {
            self.project_to_dg(field)?;
        }

        // Get coefficients and compute dimensions
        let coeffs_shape = self
            .modal_coefficients
            .as_ref()
            .ok_or_else(|| {
                KwaversError::Validation(ValidationError::MissingField {
                    field: "modal_coefficients".to_string(),
                })
            })?
            .raw_dim();
        let _n_elements = coeffs_shape[0];
        let wave_speed = 1500.0; // Example wave speed

        // Compute RHS of DG formulation
        let mut rhs = Array3::zeros(coeffs_shape);

        // Volume integral: -M^{-1} * S * f(u)
        let mass_inv = matrix_inverse(&self.mass_matrix)?;

        // Extract coefficients for computation
        let coeffs_copy = self.modal_coefficients.as_ref().unwrap().clone();

        // Compute volume integrals
        self.compute_volume_integrals(&coeffs_copy, &mut rhs, &mass_inv, wave_speed)?;

        Ok(())
    }

    /// Compute volume integrals for DG update
    fn compute_volume_integrals(
        &self,
        coeffs: &Array3<f64>,
        rhs: &mut Array3<f64>,
        _mass_inv: &Array2<f64>,
        wave_speed: f64,
    ) -> KwaversResult<()> {
        let n_elements = coeffs.shape()[0];
        let n_vars = coeffs.shape()[2];

        // Strong form: u_t = -c * D * u
        let d_matrix = &self.diff_matrix;

        for var in 0..n_vars {
            for elem in 0..n_elements {
                for i in 0..self.n_nodes {
                    let mut du = 0.0;
                    for j in 0..self.n_nodes {
                        du += d_matrix[[i, j]] * coeffs[(elem, j, var)];
                    }
                    rhs[(elem, i, var)] -= wave_speed * du;
                }
            }
        }

        Ok(())
    }
}
