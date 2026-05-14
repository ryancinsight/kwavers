//! Trait implementations for DG solver
//!
//! This module contains the implementations of standard traits
//! for the DG solver including NumericalSolver and DGOperations.
//!
//! ## Modal–Nodal DG Transform (Hesthaven & Warburton 2008, §3.1)
//!
//! **Nodal → Modal (project_to_basis):**
//! ```text
//!   c = V⁻¹ · f
//! ```
//! where `V[i,j] = P̃_j(ξ_i)` is the Vandermonde matrix evaluated at GLL nodes,
//! and V⁻¹ is precomputed at solver construction.
//!
//! **Modal → Nodal (reconstruct_from_basis):**
//! ```text
//!   f = V · c
//! ```
//!
//! **Round-trip identity:** V · V⁻¹ = I, so `reconstruct(project(f)) = f` exactly
//! for all f in the polynomial space of degree ≤ p.
//!
//! ## References
//!
//! - Hesthaven JS, Warburton T (2008). *Nodal Discontinuous Galerkin Methods*. Springer. §3.1.
//! - Kopriva DA (2009). *Implementing Spectral Methods for PDEs*. Springer. §3.4.

#[cfg(test)]
mod tests;

use super::super::traits::{DGOperations, NumericalSolver};
use super::core::DGSolver;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::{Array1, Array3};

impl NumericalSolver for DGSolver {
    fn solve(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        let mut result = field.clone();

        self.project_to_dg(&result)?;
        self.solve_step(&mut result, dt)?;

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
        // CFL condition for DG(p): dt ≤ dx / (c · (2p+1))  (Cockburn & Shu 2001 §4)
        let dx = grid.dx.min(grid.dy).min(grid.dz);
        let p = self.config.polynomial_order as f64;
        dx / (self.config.sound_speed * 2.0f64.mul_add(p, 1.0))
    }

    fn update_order(&mut self, order: usize) {
        self.config.polynomial_order = order;
    }
}

impl DGOperations for DGSolver {
    /// Upwind scalar flux for the linear advection equation `u_t + c u_x = 0`.
    ///
    /// Uses the physical wave speed from `config.sound_speed`; the upwind direction
    /// is determined by the sign of the outward normal.
    fn compute_flux(&self, left_state: f64, right_state: f64, normal: f64) -> f64 {
        if normal > 0.0 {
            self.config.sound_speed * left_state
        } else {
            self.config.sound_speed * right_state
        }
    }

    /// Project nodal values `f` to modal coefficients `c = V⁻¹ · f`.
    ///
    /// ## Layout
    ///
    /// `field` has shape `[n_elements, n_nodes, n_vars]`. For each element `e` and
    /// variable `v`, the column vector `f[e, :, v]` is multiplied by the precomputed
    /// `V⁻¹` to yield modal coefficients `c[e, :, v]`.
    ///
    /// ## Theorem (exactness)
    ///
    /// For any field `f` whose restriction to each element is a polynomial of degree
    /// ≤ p, the round-trip `reconstruct(project(f))` recovers `f` to machine
    /// precision: ‖V · (V⁻¹ · f) − f‖₂ ≤ ε_mach · κ(V) · ‖f‖₂,
    /// where κ(V) is the condition number of V.  For GLL nodes κ(V) grows only
    /// algebraically with p (Kopriva 2009 §3.4).
    ///
    /// ## Reference
    ///
    /// Hesthaven & Warburton (2008). §3.1, eq. (3.4).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    fn project_to_basis(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let n_elements = field.shape()[0];
        let n_nodes_in = field.shape()[1];
        let n_vars = field.shape()[2];

        if n_nodes_in != self.n_nodes {
            return Err(KwaversError::InvalidInput(format!(
                "project_to_basis: field has {} nodes per element, expected {}",
                n_nodes_in, self.n_nodes
            )));
        }

        let v_inv = &*self.vandermonde_inv;
        let mut coefficients = Array3::<f64>::zeros(field.raw_dim());

        for e in 0..n_elements {
            for v in 0..n_vars {
                let f_col: Array1<f64> = (0..self.n_nodes).map(|i| field[[e, i, v]]).collect();
                let c_col = v_inv.dot(&f_col);
                for i in 0..self.n_nodes {
                    coefficients[[e, i, v]] = c_col[i];
                }
            }
        }

        Ok(coefficients)
    }

    /// Reconstruct nodal values `f = V · c` from modal coefficients `c`.
    ///
    /// ## Layout
    ///
    /// `coefficients` has shape `[n_elements, n_modes, n_vars]`. For each element
    /// `e` and variable `v`, the column `c[e, :, v]` is multiplied by `V` to yield
    /// nodal values `f[e, :, v]`.
    ///
    /// ## Reference
    ///
    /// Hesthaven & Warburton (2008). §3.1, eq. (3.3).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    fn reconstruct_from_basis(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let n_elements = coefficients.shape()[0];
        let n_modes_in = coefficients.shape()[1];
        let n_vars = coefficients.shape()[2];

        if n_modes_in != self.n_nodes {
            return Err(KwaversError::InvalidInput(format!(
                "reconstruct_from_basis: coefficients have {} modes per element, expected {}",
                n_modes_in, self.n_nodes
            )));
        }

        let v = &*self.vandermonde;
        let mut field = Array3::<f64>::zeros(coefficients.raw_dim());

        for e in 0..n_elements {
            for var in 0..n_vars {
                let c_col: Array1<f64> = (0..self.n_nodes)
                    .map(|i| coefficients[[e, i, var]])
                    .collect();
                let f_col = v.dot(&c_col);
                for i in 0..self.n_nodes {
                    field[[e, i, var]] = f_col[i];
                }
            }
        }

        Ok(field)
    }
}

impl DGSolver {
    /// Get the polynomial order
    #[must_use]
    pub fn polynomial_order(&self) -> usize {
        self.config.polynomial_order
    }

    /// Get the number of nodes per element
    #[must_use]
    pub fn nodes_per_element(&self) -> usize {
        self.n_nodes
    }

    /// Get element data — returns a copy to avoid lifetime issues
    #[must_use]
    pub fn get_element_data(&self, element_id: usize) -> Option<Vec<f64>> {
        if let Some(ref coeffs) = self.modal_coefficients {
            if element_id < coeffs.shape()[0] {
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_element_data(&mut self, element_id: usize, data: &[f64]) -> KwaversResult<()> {
        if let Some(ref mut coeffs) = self.modal_coefficients {
            if element_id < coeffs.shape()[0] && data.len() == self.n_nodes {
                for (i, &value) in data.iter().enumerate() {
                    coeffs[(element_id, i, 0)] = value;
                }
                Ok(())
            } else {
                Err(KwaversError::InvalidInput(
                    "Invalid element ID or data size".to_owned(),
                ))
            }
        } else {
            Err(KwaversError::InvalidInput(
                "Modal coefficients not initialized".to_owned(),
            ))
        }
    }
}
