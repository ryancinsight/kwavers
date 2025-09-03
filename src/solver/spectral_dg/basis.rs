//! Basis function implementations for DG methods
//!
//! This module provides polynomial basis functions and related operations
//! for discontinuous Galerkin methods.

use crate::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};

/// Basis function type for DG
#[derive(Debug, Clone, Copy)]
pub enum BasisType {
    /// Legendre polynomials (modal basis)
    Legendre,
    /// Lagrange polynomials (nodal basis)
    Lagrange,
}

/// Compute Legendre polynomial and its derivative
#[must_use]
pub fn legendre_poly_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    // Recurrence relation for Legendre polynomials
    let mut p_nm2 = 1.0; // P_0
    let mut p_nm1 = x; // P_1
    let mut dp_nm2 = 0.0; // P_0'
    let mut dp_nm1 = 1.0; // P_1'

    for k in 2..=n {
        let k_f = k as f64;
        let p_n = ((2.0 * k_f - 1.0) * x * p_nm1 - (k_f - 1.0) * p_nm2) / k_f;
        let dp_n = dp_nm2 + (2.0 * k_f - 1.0) * p_nm1;

        p_nm2 = p_nm1;
        p_nm1 = p_n;
        dp_nm2 = dp_nm1;
        dp_nm1 = dp_n;
    }

    (p_nm1, dp_nm1)
}

/// Evaluate Legendre basis function
#[must_use]
pub fn legendre_basis(n: usize, x: f64) -> f64 {
    legendre_poly_deriv(n, x).0
}

/// Evaluate Lagrange basis function
#[must_use]
pub fn lagrange_basis(j: usize, x: f64, nodes: &Array1<f64>) -> f64 {
    let mut l = 1.0;
    for (i, &xi) in nodes.iter().enumerate() {
        if i != j {
            l *= (x - xi) / (nodes[j] - xi);
        }
    }
    l
}

/// Compute derivative of Lagrange basis
#[must_use]
pub fn lagrange_basis_deriv(j: usize, x: f64, nodes: &Array1<f64>) -> f64 {
    let mut dl = 0.0;

    for k in 0..nodes.len() {
        if k != j {
            let mut prod = 1.0 / (nodes[j] - nodes[k]);
            for (i, &xi) in nodes.iter().enumerate() {
                if i != j && i != k {
                    prod *= (x - xi) / (nodes[j] - xi);
                }
            }
            dl += prod;
        }
    }

    dl
}

/// Build Vandermonde matrix for basis evaluation
pub fn build_vandermonde(
    nodes: &Array1<f64>,
    poly_order: usize,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n_nodes = nodes.len();

    // For DG methods, we need a square Vandermonde matrix
    // n_nodes should equal poly_order + 1
    if n_nodes != poly_order + 1 {
        return Err(KwaversError::InvalidInput(format!(
            "Number of nodes ({}) must equal polynomial order + 1 ({})",
            n_nodes,
            poly_order + 1
        )));
    }

    let mut vandermonde = Array2::zeros((n_nodes, n_nodes));

    match basis_type {
        BasisType::Legendre => {
            for (i, &xi) in nodes.iter().enumerate() {
                for j in 0..n_nodes {
                    vandermonde[(i, j)] = legendre_basis(j, xi);
                }
            }
        }
        BasisType::Lagrange => {
            for (i, &xi) in nodes.iter().enumerate() {
                for j in 0..n_nodes {
                    vandermonde[(i, j)] = lagrange_basis(j, xi, nodes);
                }
            }
        }
    }

    Ok(vandermonde)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_legendre_polynomials() {
        // Test P_0(x) = 1
        assert_eq!(legendre_basis(0, 0.5), 1.0);

        // Test P_1(x) = x
        assert_eq!(legendre_basis(1, 0.5), 0.5);

        // Test P_2(x) = (3x^2 - 1) / 2
        let x = 0.5;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert_relative_eq!(legendre_basis(2, x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_basis() {
        let nodes = Array1::from(vec![-1.0, 0.0, 1.0]);

        // Test basis function properties
        for j in 0..nodes.len() {
            for (i, &xi) in nodes.iter().enumerate() {
                let value = lagrange_basis(j, xi, &nodes);
                if i == j {
                    assert_relative_eq!(value, 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(value, 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}
