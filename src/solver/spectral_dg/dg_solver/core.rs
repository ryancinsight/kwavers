//! Core DG solver structure and initialization
//!
//! This module contains the main DGSolver struct and its constructor,
//! separated from the implementation details for better modularity.

use super::super::basis::build_vandermonde;
use super::super::config::DGConfig;
use super::super::matrices::{
    compute_diff_matrix, compute_lift_matrix, compute_mass_matrix, compute_stiffness_matrix,
};
use super::super::quadrature::gauss_lobatto_quadrature;
use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

/// DG solver for hyperbolic conservation laws
#[derive(Debug)]
pub struct DGSolver {
    /// Configuration
    pub(super) config: DGConfig,
    /// Grid reference
    pub(super) grid: Arc<Grid>,
    /// Number of nodes per element
    pub(super) n_nodes: usize,
    /// Quadrature nodes on reference element [-1, 1]
    pub(super) xi_nodes: Arc<Array1<f64>>,
    /// Quadrature weights
    pub(super) weights: Arc<Array1<f64>>,
    /// Vandermonde matrix for basis evaluation
    pub(super) vandermonde: Arc<Array2<f64>>,
    /// Mass matrix `M_ij` = `integral(phi_i` * `phi_j`)
    pub(super) mass_matrix: Arc<Array2<f64>>,
    /// Stiffness matrix `S_ij` = `integral(phi_i` * `dphi_j/dxi`)
    pub(super) stiffness_matrix: Arc<Array2<f64>>,
    /// Differentiation matrix D = V * Dr * V^{-1}
    pub(super) diff_matrix: Arc<Array2<f64>>,
    /// Lift matrix for surface integrals
    pub(super) lift_matrix: Arc<Array2<f64>>,
    /// Modal coefficients for each element (`n_elements` x `n_nodes` x `n_vars`)
    pub(super) modal_coefficients: Option<Array3<f64>>,
}

impl DGSolver {
    /// Create a new DG solver with proper matrix initialization
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

    /// Check if modal coefficients are initialized
    pub fn has_modal_coefficients(&self) -> bool {
        self.modal_coefficients.is_some()
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
            modal_coefficients: self.modal_coefficients.clone(),
        }
    }
}
