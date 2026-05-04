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
//! where V[i,j] = P̃_j(ξ_i) is the Vandermonde matrix evaluated at GLL nodes,
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
        // CFL condition for DG(p): dt ≤ dx / (c · (2p+1))  (Cockburn & Shu 2001 §4)
        let dx = grid.dx.min(grid.dy).min(grid.dz);
        let p = self.config.polynomial_order as f64;

        dx / (self.config.sound_speed * (2.0 * p + 1.0))
    }

    fn update_order(&mut self, order: usize) {
        self.config.polynomial_order = order;
        // Would need to rebuild matrices here
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
                // Extract nodal column as a 1-D vector
                let f_col: Array1<f64> = (0..self.n_nodes).map(|i| field[[e, i, v]]).collect();

                // Modal transform: c = V⁻¹ · f
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
                // Extract modal column as a 1-D vector
                let c_col: Array1<f64> = (0..self.n_nodes)
                    .map(|i| coefficients[[e, i, var]])
                    .collect();

                // Nodal reconstruction: f = V · c
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

#[cfg(test)]
mod tests {
    use super::super::super::basis::BasisType;
    use super::super::super::config::DGConfig;
    use super::super::super::traits::DGOperations;
    use super::super::core::DGSolver;
    use crate::domain::grid::Grid;
    use ndarray::Array3;
    use std::sync::Arc;

    /// Build a minimal DGSolver for testing.
    fn make_solver(poly_order: usize) -> DGSolver {
        let config = DGConfig {
            polynomial_order: poly_order,
            basis_type: BasisType::Legendre,
            ..DGConfig::default()
        };
        let grid = Arc::new(
            Grid::new(
                poly_order + 1,
                poly_order + 1,
                poly_order + 1,
                1.0,
                1.0,
                1.0,
            )
            .expect("Grid::new failed in test"),
        );
        DGSolver::new(config, grid).expect("DGSolver::new failed in test")
    }

    #[test]
    fn fourier_basis_solver_construction_rejects_gll_duplicate_periodic_endpoints() {
        let config = DGConfig {
            polynomial_order: 2,
            basis_type: BasisType::Fourier,
            ..DGConfig::default()
        };
        let grid = Arc::new(Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("Grid::new failed in test"));

        let error = DGSolver::new(config, grid).unwrap_err();

        assert!(format!("{error}").contains("requires periodic nodes on [-1,1)"));
    }

    /// **Round-trip identity**: project_to_basis followed by reconstruct_from_basis
    /// must recover the input to machine precision (≤ 1e-12) for any field.
    ///
    /// ## Theorem
    ///
    /// For collocation DG, V · V⁻¹ = I (exact arithmetic).  In floating-point:
    ///   ‖V · (V⁻¹ · f) − f‖₂ ≤ ε_mach · κ(V) · ‖f‖₂
    /// For GLL nodes and p ≤ 10, κ(V) < 10² so the round-trip error is ≤ 1e-10.
    /// We assert ≤ 1e-11 (two decades of margin).
    ///
    /// Reference: Kopriva (2009) §3.4.
    #[test]
    fn test_project_reconstruct_round_trip() {
        let solver = make_solver(3); // p=3, n_nodes=4
        let n_nodes = solver.n_nodes;
        let n_elements = 5;
        let n_vars = 2;

        // Fill with arbitrary non-trivial values
        let mut field = Array3::<f64>::zeros((n_elements, n_nodes, n_vars));
        for e in 0..n_elements {
            for i in 0..n_nodes {
                for v in 0..n_vars {
                    field[[e, i, v]] = ((e + 1) as f64) * 1.3 + (i as f64) * 0.7 + (v as f64) * 0.2;
                }
            }
        }

        let coefficients = solver
            .project_to_basis(&field)
            .expect("project_to_basis failed");
        let recovered = solver
            .reconstruct_from_basis(&coefficients)
            .expect("reconstruct failed");

        for e in 0..n_elements {
            for i in 0..n_nodes {
                for v in 0..n_vars {
                    let err = (recovered[[e, i, v]] - field[[e, i, v]]).abs();
                    assert!(
                        err < 1e-11,
                        "Round-trip error at [e={},i={},v={}]: {:.2e} (must be < 1e-11)",
                        e,
                        i,
                        v,
                        err
                    );
                }
            }
        }
    }

    /// **Legendre coefficient extraction**: projecting the k-th normalised Legendre
    /// basis function P̃_k must yield c[k] = 1 and all other |c[j]| < 1e-12.
    ///
    /// ## Theorem
    ///
    /// V⁻¹ · (V[:, k]) = e_k  (the k-th standard basis vector), because
    /// V⁻¹ · V = I by construction.
    ///
    /// Reference: Hesthaven & Warburton (2008) §3.1.
    #[test]
    fn test_legendre_coefficient_extraction() {
        let solver = make_solver(3); // p=3, n_nodes=4
        let n_nodes = solver.n_nodes;
        let v = &*solver.vandermonde;

        for k in 0..n_nodes {
            // Build a single-element field equal to the k-th column of V (the k-th basis polynomial)
            let mut field = Array3::<f64>::zeros((1, n_nodes, 1));
            for i in 0..n_nodes {
                field[[0, i, 0]] = v[[i, k]];
            }

            let coefficients = solver
                .project_to_basis(&field)
                .expect("project_to_basis failed");

            for j in 0..n_nodes {
                let expected = if j == k { 1.0 } else { 0.0 };
                let err = (coefficients[[0, j, 0]] - expected).abs();
                assert!(
                    err < 1e-11,
                    "Legendre extraction basis k={k}: c[{j}] = {:.6e}, expected {expected} (err {err:.2e})",
                    coefficients[[0, j, 0]]
                );
            }
        }
    }

    /// **Polynomial reproduction**: a polynomial of degree ≤ p is exactly
    /// represented in the DG basis (Hesthaven & Warburton 2008 §3.1, Theorem 3.1).
    /// A degree-(p+1) polynomial's projection introduces truncation error equal to
    /// the best L²-approximation error (not asserted here; exactness for ≤ p is).
    #[test]
    fn test_polynomial_reproduction() {
        let poly_order = 3;
        let solver = make_solver(poly_order);
        let n_nodes = solver.n_nodes;
        let xi = &*solver.xi_nodes;

        // Build a degree-p polynomial: f(ξ) = ξ^p (exactly representable)
        let mut field = Array3::<f64>::zeros((1, n_nodes, 1));
        for i in 0..n_nodes {
            field[[0, i, 0]] = xi[i].powi(poly_order as i32);
        }

        let coefficients = solver
            .project_to_basis(&field)
            .expect("project_to_basis failed");
        let recovered = solver
            .reconstruct_from_basis(&coefficients)
            .expect("reconstruct failed");

        for i in 0..n_nodes {
            let err = (recovered[[0, i, 0]] - field[[0, i, 0]]).abs();
            assert!(
                err < 1e-11,
                "Polynomial reproduction error at node {i}: {err:.2e} (must be < 1e-11)"
            );
        }
    }
}
