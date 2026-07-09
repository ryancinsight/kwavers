//! Core DG solver structure and initialization
//!
//! This module contains the main DGSolver struct and its constructor,
//! separated from the implementation details for better modularity.

use super::super::basis::build_vandermonde;
use super::super::basis::BasisType;
use super::super::config::DGConfig;
use super::super::matrices::{
    compute_diff_matrix, compute_lift_matrix, compute_mass_matrix, compute_stiffness_matrix,
    matrix_inverse,
};
use super::super::quadrature::{fourier_periodic_nodes, gauss_lobatto_quadrature};
use super::topology::CoefficientLayout;
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, NumericalError};
use kwavers_grid::Grid;
use leto::{
    Array1,
    Array2,
    Array3,
};
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
    /// Vandermonde matrix V[i,j] = P̃_j(ξ_i) for basis evaluation
    pub(super) vandermonde: Arc<Array2<f64>>,
    /// Inverse Vandermonde matrix V⁻¹ for modal projection (c = V⁻¹·f).
    ///
    /// Precomputed at construction; well-conditioned for GLL nodes
    /// (Kopriva 2009 §3.4).
    pub(super) vandermonde_inv: Arc<Array2<f64>>,
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
    /// Layout contract for the current coefficient tensor.
    pub(super) coefficient_layout: CoefficientLayout,
    /// SSP-RK original-state register, allocated to the modal coefficient shape.
    pub(super) rk_original: Array3<f64>,
    /// SSP-RK stage register, allocated to the modal coefficient shape.
    pub(super) rk_stage: Array3<f64>,
    /// Spatial RHS register, allocated to the modal coefficient shape.
    pub(super) rk_rhs: Array3<f64>,
}

impl DGSolver {
    /// Create a new DG solver with proper matrix initialization
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: DGConfig, grid: Arc<Grid>) -> KwaversResult<Self> {
        if config.basis_type == BasisType::Fourier {
            return Err(KwaversError::Numerical(NumericalError::UnsupportedOperation {
                operation: "DGSolver::new".to_owned(),
                reason: "Fourier DG requires periodic nodes on [-1,1); the current nodal DG constructor uses GLL nodes with duplicate periodic endpoints".to_owned(),
            }));
        }

        let n_nodes = config.polynomial_order + 1;

        // Generate quadrature nodes and weights
        let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;

        // Build Vandermonde matrix V[i,j] = P̃_j(ξ_i)
        let vandermonde = build_vandermonde(&xi_nodes, config.polynomial_order, config.basis_type)?;

        // Precompute V⁻¹ for modal projection (c = V⁻¹·f).
        // GLL nodes make V well-conditioned; Gauss-Jordan inversion exact to
        // machine precision for the polynomial orders used in acoustic DG.
        let vandermonde_inv = matrix_inverse(&vandermonde)?;

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
            vandermonde_inv: Arc::new(vandermonde_inv),
            mass_matrix: Arc::new(mass_matrix),
            stiffness_matrix: Arc::new(stiffness_matrix),
            diff_matrix: Arc::new(diff_matrix),
            lift_matrix: Arc::new(lift_matrix),
            modal_coefficients: None,
            coefficient_layout: CoefficientLayout::Line1D,
            rk_original: Array3::zeros((0, n_nodes, 0)),
            rk_stage: Array3::zeros((0, n_nodes, 0)),
            rk_rhs: Array3::zeros((0, n_nodes, 0)),
        })
    }

    /// Create a DG solver using the real Fourier basis on equispaced periodic nodes.
    ///
    /// ## Contract
    ///
    /// Nodes `x_j = -1 + 2j/N`, `j = 0,...,N-1` lie on `[-1, 1)` and satisfy the
    /// Fourier periodicity constraint (no duplicate endpoints). Weights are `w_j = 2/N`
    /// so that the N-point trapezoidal rule integrates real trigonometric polynomials of
    /// degree ≤ `⌊N/2⌋` exactly (Davis & Rabinowitz 1984, §2.9). The differentiation
    /// matrix is constructed via the spectral derivative formula:
    /// ```text
    ///   d/dx sin(kπ(x+1)) = kπ cos(kπ(x+1))
    ///   d/dx cos(kπ(x+1)) = -kπ sin(kπ(x+1))
    /// ```
    /// The lift matrix is zero for a purely periodic element: there is no net boundary
    /// flux at the periodic endpoints within a single element, so face corrections do
    /// not project onto interior DOFs. Inter-element coupling for multi-element Fourier
    /// DG is handled at the orchestrator level.
    ///
    /// `config.basis_type` is overridden to `BasisType::Fourier`; all other fields apply.
    ///
    /// # Errors
    /// Returns [`KwaversError::Config`] when `polynomial_order + 1 < 2`.
    /// Propagates matrix construction errors.
    pub fn new_fourier(mut config: DGConfig, grid: Arc<Grid>) -> KwaversResult<Self> {
        config.basis_type = BasisType::Fourier;
        let n_nodes = config.polynomial_order + 1;

        // Equispaced periodic quadrature — exact for the real Fourier basis.
        let (xi_nodes, weights) = fourier_periodic_nodes(n_nodes)?;

        // Vandermonde matrix using real Fourier columns.
        let vandermonde =
            build_vandermonde(&xi_nodes, config.polynomial_order, BasisType::Fourier)?;

        // V⁻¹ for modal projection; well-conditioned for equispaced nodes and N ≥ 2.
        let vandermonde_inv = matrix_inverse(&vandermonde)?;

        // Diagonal mass matrix: M_ii = w_i = 2/N.
        let mass_matrix = compute_mass_matrix(&vandermonde, &weights)?;

        // Stiffness and differentiation matrices from the Fourier spectral derivative.
        let stiffness_matrix =
            compute_stiffness_matrix(&vandermonde, &xi_nodes, &weights, BasisType::Fourier)?;
        let diff_matrix = compute_diff_matrix(&vandermonde, &xi_nodes, BasisType::Fourier)?;

        // Zero lift: no net boundary flux within a periodic element.
        let lift_matrix = Array2::zeros((n_nodes, 2));

        Ok(Self {
            config,
            grid,
            n_nodes,
            xi_nodes: Arc::new(xi_nodes),
            weights: Arc::new(weights),
            vandermonde: Arc::new(vandermonde),
            vandermonde_inv: Arc::new(vandermonde_inv),
            mass_matrix: Arc::new(mass_matrix),
            stiffness_matrix: Arc::new(stiffness_matrix),
            diff_matrix: Arc::new(diff_matrix),
            lift_matrix: Arc::new(lift_matrix),
            modal_coefficients: None,
            coefficient_layout: CoefficientLayout::Line1D,
            rk_original: Array3::zeros((0, n_nodes, 0)),
            rk_stage: Array3::zeros((0, n_nodes, 0)),
            rk_rhs: Array3::zeros((0, n_nodes, 0)),
        })
    }

    /// Check if modal coefficients are initialized
    #[must_use]
    pub fn has_modal_coefficients(&self) -> bool {
        self.modal_coefficients.is_some()
    }

    /// Solver configuration.
    #[must_use]
    pub fn config(&self) -> DGConfig {
        self.config
    }
}

impl Clone for DGSolver {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            grid: Arc::clone(&self.grid),
            n_nodes: self.n_nodes,
            xi_nodes: Arc::clone(&self.xi_nodes),
            weights: Arc::clone(&self.weights),
            vandermonde: Arc::clone(&self.vandermonde),
            vandermonde_inv: Arc::clone(&self.vandermonde_inv),
            mass_matrix: Arc::clone(&self.mass_matrix),
            stiffness_matrix: Arc::clone(&self.stiffness_matrix),
            diff_matrix: Arc::clone(&self.diff_matrix),
            lift_matrix: Arc::clone(&self.lift_matrix),
            modal_coefficients: self.modal_coefficients.clone(),
            coefficient_layout: self.coefficient_layout,
            rk_original: self.rk_original.clone(),
            rk_stage: self.rk_stage.clone(),
            rk_rhs: self.rk_rhs.clone(),
        }
    }
}
