//! Matrix operations for DG methods
//!
//! This module provides matrix computations specific to discontinuous
//! Galerkin methods, including mass, stiffness, and differentiation matrices.

use crate::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};

use super::basis::{lagrange_basis_deriv, legendre_basis, BasisType};

/// Compute mass matrix M_ij = integral(phi_i * phi_j)
pub fn compute_mass_matrix(
    vandermonde: &Array2<f64>,
    weights: &Array1<f64>,
) -> KwaversResult<Array2<f64>> {
    let n = vandermonde.nrows();
    let mut mass = Array2::zeros((n, n));

    // M = V^T * W * V where W is diagonal weight matrix
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                mass[(i, j)] += vandermonde[(k, i)] * weights[k] * vandermonde[(k, j)];
            }
        }
    }

    Ok(mass)
}

/// Compute stiffness matrix S_ij = integral(phi_i * dphi_j/dxi)
pub fn compute_stiffness_matrix(
    vandermonde: &Array2<f64>,
    nodes: &Array1<f64>,
    weights: &Array1<f64>,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n = nodes.len();
    let p = vandermonde.ncols() - 1; // polynomial order
    let mut stiffness = Array2::zeros((p + 1, p + 1));

    // Build derivative Vandermonde matrix
    let mut vandermonde_deriv = Array2::zeros((n, p + 1));

    match basis_type {
        BasisType::Legendre => {
            for (i, &xi) in nodes.iter().enumerate() {
                for j in 0..=p {
                    if j == 0 {
                        vandermonde_deriv[(i, j)] = 0.0;
                    } else {
                        // Derivative of Legendre polynomial
                        let mut dp = 0.0;
                        if j == 1 {
                            dp = 1.0;
                        } else {
                            // Use recurrence for derivative
                            dp = j as f64 * legendre_basis(j - 1, xi)
                                + xi * vandermonde_deriv[(i, j - 1)];
                        }
                        vandermonde_deriv[(i, j)] = dp;
                    }
                }
            }
        }
        BasisType::Lagrange => {
            for (i, &xi) in nodes.iter().enumerate() {
                for j in 0..n.min(p + 1) {
                    vandermonde_deriv[(i, j)] = lagrange_basis_deriv(j, xi, nodes);
                }
            }
        }
    }

    // S = V^T * W * V'
    for i in 0..=p {
        for j in 0..=p {
            for k in 0..n {
                stiffness[(i, j)] += vandermonde[(k, i)] * weights[k] * vandermonde_deriv[(k, j)];
            }
        }
    }

    Ok(stiffness)
}

/// Compute differentiation matrix D = V * Dr * V^{-1}
pub fn compute_diff_matrix(
    vandermonde: &Array2<f64>,
    nodes: &Array1<f64>,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n = nodes.len();
    let mut diff_matrix = Array2::zeros((n, n));

    match basis_type {
        BasisType::Lagrange => {
            // For Lagrange basis, differentiation matrix is direct
            for i in 0..n {
                for j in 0..n {
                    diff_matrix[(i, j)] = lagrange_basis_deriv(j, nodes[i], nodes);
                }
            }
        }
        BasisType::Legendre => {
            // For modal basis, use D = V * Dr * V^{-1}
            // First compute V^{-1}
            let v_inv = matrix_inverse(vandermonde)?;

            // Build Dr in modal space
            let p = vandermonde.ncols() - 1;
            let mut dr_modal = Array2::zeros((p + 1, p + 1));

            // Derivative in modal space for Legendre
            for i in 0..p {
                dr_modal[(i, i + 1)] = (2 * i + 1) as f64;
            }

            // D = V * Dr * V^{-1}
            let temp = vandermonde.dot(&dr_modal);
            diff_matrix = temp.dot(&v_inv);
        }
    }

    Ok(diff_matrix)
}

/// Compute lift matrix for surface integrals
pub fn compute_lift_matrix(
    mass_matrix: &Array2<f64>,
    n_nodes: usize,
) -> KwaversResult<Array2<f64>> {
    // Lift matrix maps surface integrals to volume
    // For 1D, it's M^{-1} * E where E extracts boundary values

    let mass_inv = matrix_inverse(mass_matrix)?;
    let mut lift = Array2::zeros((n_nodes, 2)); // 2 boundaries in 1D

    // Extract matrix: E[0,:] = [1, 0, ..., 0], E[1,:] = [0, ..., 0, 1]
    lift[(0, 0)] = 1.0;
    lift[(n_nodes - 1, 1)] = 1.0;

    // Lift = M^{-1} * E
    let result = mass_inv.dot(&lift);

    Ok(result)
}

/// Matrix inversion using LU decomposition
pub fn matrix_inverse(a: &Array2<f64>) -> KwaversResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(KwaversError::InvalidInput(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    // Create augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
            if i == j {
                aug[(i, n + j)] = 1.0;
            }
        }
    }

    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if aug[(i, k)].abs() > aug[(max_row, k)].abs() {
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..(2 * n) {
                let temp = aug[(k, j)];
                aug[(k, j)] = aug[(max_row, j)];
                aug[(max_row, j)] = temp;
            }
        }

        // Check for singularity with better tolerance
        if aug[(k, k)].abs() < 1e-12 {
            return Err(KwaversError::NumericalError(format!(
                "Matrix is singular or nearly singular at pivot {}: value = {}",
                k,
                aug[(k, k)]
            )));
        }

        // Forward elimination
        for i in (k + 1)..n {
            let factor = aug[(i, k)] / aug[(k, k)];
            for j in k..(2 * n) {
                aug[(i, j)] -= factor * aug[(k, j)];
            }
        }
    }

    // Back substitution
    for k in (0..n).rev() {
        for j in n..(2 * n) {
            aug[(k, j)] /= aug[(k, k)];
        }
        aug[(k, k)] = 1.0;

        for i in 0..k {
            let factor = aug[(i, k)];
            for j in n..(2 * n) {
                aug[(i, j)] -= factor * aug[(k, j)];
            }
            aug[(i, k)] = 0.0;
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = aug[(i, n + j)];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matrix_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 2.0, 1.0]).unwrap();
        let inv = matrix_inverse(&a).unwrap();

        // Check A * A^{-1} = I
        let identity = a.dot(&inv);
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mass_matrix_properties() {
        // Mass matrix should be symmetric and positive definite
        let nodes = Array1::from(vec![-1.0, 0.0, 1.0]);
        let weights = Array1::from(vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]);
        let vandermonde = Array2::eye(3);

        let mass = compute_mass_matrix(&vandermonde, &weights).unwrap();

        // Check symmetry
        for i in 0..3 {
            for j in i + 1..3 {
                assert_relative_eq!(mass[(i, j)], mass[(j, i)], epsilon = 1e-10);
            }
        }
    }
}
