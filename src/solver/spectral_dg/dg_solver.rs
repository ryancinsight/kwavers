//! Discontinuous Galerkin (DG) solver implementation
//!
//! This module implements a DG method for solving hyperbolic conservation laws
//! with shock-capturing capabilities for handling discontinuities.
//!
//! The implementation is split into focused submodules following GRASP principles:
//! - `core`: Main DGSolver struct and constructor  
//! - `projection`: Field projection onto DG basis
//! - `solver_ops`: Core solver operations and time stepping
//! - `trait_impls`: Trait implementations
//!
//! # References
//!
//! - Hesthaven, J. S., & Warburton, T. (2008). "Nodal discontinuous Galerkin methods"
//! - Cockburn, B., & Shu, C. W. (2001). "Runge-Kutta discontinuous Galerkin methods"

// Import submodules
mod core;
mod projection;
mod solver_ops;
mod trait_impls;

// Re-export main types for backwards compatibility
pub use core::DGSolver;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::spectral_dg::basis::build_vandermonde;
    use crate::solver::spectral_dg::config::DGConfig;
    use crate::solver::spectral_dg::matrices::compute_mass_matrix;
    use crate::solver::spectral_dg::quadrature::gauss_lobatto_quadrature;
    use crate::grid::Grid;
    use std::sync::Arc;

    #[test]
    fn test_dg_solver_creation() {
        let grid = Arc::new(Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap());
        let config = DGConfig::default();

        // Debug: Check intermediate steps
        let n_nodes = config.polynomial_order + 1;
        println!(
            "Creating DG solver with {} nodes (order {})",
            n_nodes, config.polynomial_order
        );

        // Test quadrature
        if let Ok((nodes, weights)) = gauss_lobatto_quadrature(n_nodes) {
            println!("Quadrature nodes: {:?}", nodes);
            println!("Quadrature weights: {:?}", weights);

            // Test Vandermonde matrix
            if let Ok(v) = build_vandermonde(&nodes, config.polynomial_order, config.basis_type) {
                println!("Vandermonde matrix shape: {:?}", v.shape());

                // Test mass matrix
                if let Ok(m) = compute_mass_matrix(&v, &weights) {
                    println!("Mass matrix shape: {:?}", m.shape());
                    println!(
                        "Mass matrix diagonal: {:?}",
                        (0..n_nodes).map(|i| m[(i, i)]).collect::<Vec<_>>()
                    );

                    // Check if matrix is singular by looking at diagonal
                    let min_diag = (0..n_nodes)
                        .map(|i| m[(i, i)].abs())
                        .fold(f64::INFINITY, f64::min);
                    println!("Minimum diagonal element: {:.2e}", min_diag);

                    // Check determinant approximation (product of diagonal for triangular)
                    let det_approx = (0..n_nodes).map(|i| m[(i, i)]).product::<f64>();
                    println!("Diagonal product (det approximation): {:.2e}", det_approx);
                }
            }
        }

        let solver = DGSolver::new(config, grid);
        if let Err(e) = &solver {
            eprintln!("DGSolver creation failed: {:?}", e);
        }
        assert!(solver.is_ok());
    }

    #[test]
    fn test_dg_solver_properties() {
        let grid = Arc::new(Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap());
        let config = DGConfig::default();
        let solver = DGSolver::new(config, grid).unwrap();

        assert_eq!(solver.polynomial_order(), config.polynomial_order);
        assert_eq!(solver.nodes_per_element(), config.polynomial_order + 1);
        assert!(!solver.has_modal_coefficients());
    }
}