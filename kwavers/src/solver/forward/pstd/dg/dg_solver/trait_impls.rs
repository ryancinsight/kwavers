//! Trait implementations for DG solver
//!
//! This module contains the implementations of standard traits
//! for the DG solver including NumericalSolver and DGOperations.

use super::super::traits::{DGOperations, NumericalSolver};
use super::core::DGSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;

impl NumericalSolver for DGSolver {
    fn solve(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        // Use self directly since we have mutable access
        let mut result = field.clone();

        // Apply DG method only in regions marked by mask
        self.project_to_dg(&result)?;
        self.solve_step(&mut result, dt)?;

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

impl DGOperations for DGSolver {
    fn compute_flux(&self, left_state: f64, right_state: f64, normal: f64) -> f64 {
        // Simple upwind flux for scalar conservation law
        let wave_speed = 1500.0; // Example wave speed
        if normal > 0.0 {
            wave_speed * left_state
        } else {
            wave_speed * right_state
        }
    }

    fn project_to_basis(&self, _field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Full DG projection requires: û = M^(-1) ∫ u(x) φ_i(x) dx
        // where φ_i are Legendre/Lagrange basis functions and M is the mass matrix.
        // Deferred to Sprint 122+ for full discontinuous Galerkin solver expansion.
        Err(KwaversError::NotImplemented(
            "DG project_to_basis: Legendre polynomial projection with mass matrix inversion not yet implemented".to_string(),
        ))
    }

    fn reconstruct_from_basis(&self, _coefficients: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Full DG reconstruction: u(x) = Σ û_i φ_i(x) evaluated at quadrature points.
        // Deferred to Sprint 122+ for full discontinuous Galerkin solver expansion.
        Err(KwaversError::NotImplemented(
            "DG reconstruct_from_basis: polynomial expansion evaluation not yet implemented".to_string(),
        ))
    }
}

impl DGSolver {
    /// Get the polynomial order
    pub fn polynomial_order(&self) -> usize {
        self.config.polynomial_order
    }

    /// Get the number of nodes per element
    pub fn nodes_per_element(&self) -> usize {
        self.n_nodes
    }

    /// Get element data - returns a copy to avoid lifetime issues
    pub fn get_element_data(&self, element_id: usize) -> Option<Vec<f64>> {
        if let Some(ref coeffs) = self.modal_coefficients {
            if element_id < coeffs.shape()[0] {
                // Return copy for single variable (index 0)
                let mut data = Vec::with_capacity(self.n_nodes);
                for node in 0..self.n_nodes {
                    data.push(coeffs[(element_id, node, 0)]);
                }
                Some(data)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Set element data
    pub fn set_element_data(&mut self, element_id: usize, data: &[f64]) -> KwaversResult<()> {
        if let Some(ref mut coeffs) = self.modal_coefficients {
            if element_id < coeffs.shape()[0] && data.len() == self.n_nodes {
                // Set data for single variable (index 0)
                for (i, &value) in data.iter().enumerate() {
                    coeffs[(element_id, i, 0)] = value;
                }
                Ok(())
            } else {
                Err(crate::core::error::KwaversError::InvalidInput(
                    "Invalid element ID or data size".to_string(),
                ))
            }
        } else {
            Err(crate::core::error::KwaversError::InvalidInput(
                "Modal coefficients not initialized".to_string(),
            ))
        }
    }
}
