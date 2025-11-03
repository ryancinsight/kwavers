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

/// Compute Gauss-Lobatto-Legendre (GLL) quadrature nodes and weights
///
/// GLL nodes include the endpoints [-1, 1] and are roots of (1-x²)P'_n(x)
/// where P'_n is the derivative of the Legendre polynomial of order n.
///
/// # Arguments
/// * `n` - Number of quadrature points (polynomial order + 1)
///
/// # Returns
/// Tuple of (nodes, weights) both as `Array1<f64>`
///
/// # References
/// - Karniadakis & Sherwin (2005), "Spectral/hp Element Methods for CFD"
/// - Hesthaven & Warburton (2008), "Nodal Discontinuous Galerkin Methods"
pub fn gauss_lobatto_legendre_nodes(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
    if n < 2 {
        return Err(KwaversError::InvalidInput(
            "GLL quadrature requires at least 2 points".to_string(),
        ));
    }

    let mut nodes = Array1::zeros(n);
    let mut weights = Array1::zeros(n);

    // Endpoints are always -1 and 1
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;

    if n == 2 {
        weights[0] = 1.0;
        weights[1] = 1.0;
        return Ok((nodes, weights));
    }

    // Interior nodes are roots of (1-x²)P'_(n-1)(x)
    // Use Newton-Raphson iteration
    for j in 1..n - 1 {
        // Initial guess: Chebyshev nodes
        let mut x = -(std::f64::consts::PI * (j as f64) / (n as f64 - 1.0)).cos();

        // Newton-Raphson iteration
        for _ in 0..50 {
            let (_p, dp) = legendre_poly_deriv(n - 1, x);
            let f = (1.0 - x * x) * dp; // (1-x²)P'_(n-1)(x)
            let df = -2.0 * x * dp + (1.0 - x * x) * second_derivative_legendre(n - 1, x);

            let dx = f / df;
            x -= dx;

            if dx.abs() < 1e-15 {
                break;
            }
        }

        nodes[j] = x;
    }

    // Compute weights: w_j = 2 / (n(n-1) * [P_(n-1)(x_j)]²)
    for j in 0..n {
        let p = legendre_basis(n - 1, nodes[j]);
        weights[j] = 2.0 / ((n as f64) * (n as f64 - 1.0) * p * p);
    }

    Ok((nodes, weights))
}

/// Compute second derivative of Legendre polynomial
/// Uses the relation: (1-x²)P''_n - 2xP'_n + n(n+1)P_n = 0
#[must_use]
fn second_derivative_legendre(n: usize, x: f64) -> f64 {
    let (p, dp) = legendre_poly_deriv(n, x);
    if (1.0 - x * x).abs() < 1e-14 {
        // Near endpoints, use limit
        (n as f64) * (n as f64 + 1.0) * p / 2.0
    } else {
        (2.0 * x * dp - (n as f64) * (n as f64 + 1.0) * p) / (1.0 - x * x)
    }
}

/// Build mass matrix for DG method
///
/// The mass matrix M_ij = ∫ φ_i(x) φ_j(x) dx where φ are basis functions
///
/// # Arguments
/// * `nodes` - Quadrature nodes
/// * `weights` - Quadrature weights
/// * `poly_order` - Polynomial order
/// * `basis_type` - Type of basis functions
///
/// # Returns
/// Mass matrix as `Array2<f64>`
///
/// # References
/// - Hesthaven & Warburton (2008), "Nodal Discontinuous Galerkin Methods", Section 3.2
pub fn build_mass_matrix(
    nodes: &Array1<f64>,
    weights: &Array1<f64>,
    poly_order: usize,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n = poly_order + 1;
    let mut mass = Array2::zeros((n, n));

    // M_ij = ∫ φ_i(x) φ_j(x) dx ≈ Σ w_k φ_i(x_k) φ_j(x_k)
    for i in 0..n {
        for j in 0..n {
            let mut integral = 0.0;
            for (&x_k, &w_k) in nodes.iter().zip(weights.iter()) {
                let phi_i = match basis_type {
                    BasisType::Legendre => legendre_basis(i, x_k),
                    BasisType::Lagrange => lagrange_basis(i, x_k, nodes),
                };
                let phi_j = match basis_type {
                    BasisType::Legendre => legendre_basis(j, x_k),
                    BasisType::Lagrange => lagrange_basis(j, x_k, nodes),
                };
                integral += w_k * phi_i * phi_j;
            }
            mass[(i, j)] = integral;
        }
    }

    Ok(mass)
}

/// Build stiffness matrix for DG method
///
/// The stiffness matrix S_ij = ∫ φ_i(x) φ'_j(x) dx where φ' is the derivative
///
/// # Arguments
/// * `nodes` - Quadrature nodes
/// * `weights` - Quadrature weights
/// * `poly_order` - Polynomial order
/// * `basis_type` - Type of basis functions
///
/// # Returns
/// Stiffness matrix as `Array2<f64>`
///
/// # References
/// - Hesthaven & Warburton (2008), "Nodal Discontinuous Galerkin Methods", Section 3.2
pub fn build_stiffness_matrix(
    nodes: &Array1<f64>,
    weights: &Array1<f64>,
    poly_order: usize,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n = poly_order + 1;
    let mut stiffness = Array2::zeros((n, n));

    // S_ij = ∫ φ_i(x) φ'_j(x) dx ≈ Σ w_k φ_i(x_k) φ'_j(x_k)
    for i in 0..n {
        for j in 0..n {
            let mut integral = 0.0;
            for (&x_k, &w_k) in nodes.iter().zip(weights.iter()) {
                let phi_i = match basis_type {
                    BasisType::Legendre => legendre_basis(i, x_k),
                    BasisType::Lagrange => lagrange_basis(i, x_k, nodes),
                };
                let dphi_j = match basis_type {
                    BasisType::Legendre => legendre_poly_deriv(j, x_k).1,
                    BasisType::Lagrange => lagrange_basis_deriv(j, x_k, nodes),
                };
                integral += w_k * phi_i * dphi_j;
            }
            stiffness[(i, j)] = integral;
        }
    }

    Ok(stiffness)
}

/// Build differentiation matrix for DG method
///
/// The differentiation matrix D relates nodal values to their derivatives:
/// du/dx|_{x_i} = Σ_j D_ij u_j
///
/// # Arguments
/// * `nodes` - Quadrature nodes
/// * `poly_order` - Polynomial order
/// * `basis_type` - Type of basis functions
///
/// # Returns
/// Differentiation matrix as `Array2<f64>`
pub fn build_differentiation_matrix(
    nodes: &Array1<f64>,
    poly_order: usize,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n = poly_order + 1;
    let mut diff = Array2::zeros((n, n));

    // D_ij = φ'_j(x_i)
    for i in 0..n {
        for j in 0..n {
            diff[(i, j)] = match basis_type {
                BasisType::Legendre => legendre_poly_deriv(j, nodes[i]).1,
                BasisType::Lagrange => lagrange_basis_deriv(j, nodes[i], nodes),
            };
        }
    }

    Ok(diff)
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

    #[test]
    fn test_gll_nodes_n2() {
        let (nodes, weights) = gauss_lobatto_legendre_nodes(2).unwrap();

        // Check endpoints
        assert_relative_eq!(nodes[0], -1.0, epsilon = 1e-14);
        assert_relative_eq!(nodes[1], 1.0, epsilon = 1e-14);

        // Check weights
        assert_relative_eq!(weights[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(weights[1], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_gll_nodes_n3() {
        let (nodes, weights) = gauss_lobatto_legendre_nodes(3).unwrap();

        // Check endpoints
        assert_relative_eq!(nodes[0], -1.0, epsilon = 1e-14);
        assert_relative_eq!(nodes[2], 1.0, epsilon = 1e-14);

        // Middle node should be 0
        assert_relative_eq!(nodes[1], 0.0, epsilon = 1e-14);

        // Check weights sum to 2 (integral over [-1,1])
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mass_matrix_legendre() {
        let n = 3;
        let (nodes, weights) = gauss_lobatto_legendre_nodes(n).unwrap();
        let mass = build_mass_matrix(&nodes, &weights, n - 1, BasisType::Legendre).unwrap();

        // Mass matrix should be symmetric
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(mass[(i, j)], mass[(j, i)], epsilon = 1e-12);
            }
        }

        // Diagonal should be positive
        for i in 0..n {
            assert!(mass[(i, i)] > 0.0);
        }
    }

    #[test]
    fn test_mass_matrix_lagrange() {
        let n = 3;
        let (nodes, weights) = gauss_lobatto_legendre_nodes(n).unwrap();
        let mass = build_mass_matrix(&nodes, &weights, n - 1, BasisType::Lagrange).unwrap();

        // For Lagrange basis with GLL nodes, mass matrix should be diagonal
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert!(mass[(i, j)] > 0.0);
                } else {
                    assert_relative_eq!(mass[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_stiffness_matrix() {
        let n = 3;
        let (nodes, weights) = gauss_lobatto_legendre_nodes(n).unwrap();
        let stiffness =
            build_stiffness_matrix(&nodes, &weights, n - 1, BasisType::Legendre).unwrap();

        // Stiffness matrix exists and is finite
        for i in 0..n {
            for j in 0..n {
                assert!(stiffness[(i, j)].is_finite());
            }
        }

        // Stiffness matrix should have non-zero elements
        let has_nonzero = (0..n).any(|i| (0..n).any(|j| stiffness[(i, j)].abs() > 1e-12));
        assert!(
            has_nonzero,
            "Stiffness matrix should have non-zero elements"
        );
    }

    #[test]
    fn test_differentiation_matrix() {
        let n = 3;
        let (nodes, _) = gauss_lobatto_legendre_nodes(n).unwrap();
        let diff = build_differentiation_matrix(&nodes, n - 1, BasisType::Lagrange).unwrap();

        // Test differentiation of a linear function: f(x) = x
        let f = nodes.clone();
        let df = diff.dot(&f);

        // Derivative should be 1 everywhere
        for i in 0..n {
            assert_relative_eq!(df[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_differentiation_quadratic() {
        let n = 4;
        let (nodes, _) = gauss_lobatto_legendre_nodes(n).unwrap();
        let diff = build_differentiation_matrix(&nodes, n - 1, BasisType::Lagrange).unwrap();

        // Test differentiation of quadratic: f(x) = x²
        let f: Array1<f64> = nodes.iter().map(|&x| x * x).collect();
        let df = diff.dot(&f);

        // Derivative should be 2x
        for i in 0..n {
            let expected = 2.0 * nodes[i];
            assert_relative_eq!(df[i], expected, epsilon = 1e-9);
        }
    }
}
