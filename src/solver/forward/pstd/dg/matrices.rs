//! Matrix computation utilities for DG methods

use super::basis::BasisType;
use crate::core::error::KwaversResult;
use crate::core::error::{KwaversError, NumericalError};
use ndarray::{Array1, Array2};

/// Compute mass matrix using quadrature
/// M_ij = integral(phi_i * phi_j)
pub fn compute_mass_matrix(
    _vandermonde: &Array2<f64>,
    weights: &Array1<f64>,
) -> KwaversResult<Array2<f64>> {
    // With GLL quadrature, M is diagonal with entries equal to weights.

    let n = weights.len();
    let mut m = Array2::zeros((n, n));

    for i in 0..n {
        m[[i, i]] = weights[i];
    }

    Ok(m)
}

/// Compute stiffness matrix
/// S_ij = integral(phi_i * phi'_j)
pub fn compute_stiffness_matrix(
    vandermonde: &Array2<f64>,
    nodes: &Array1<f64>,
    weights: &Array1<f64>,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    // S = M * D.

    // First compute D (differentiation matrix).
    let d = compute_diff_matrix(vandermonde, nodes, basis_type)?;

    // Then S = M * D.
    // Since M is diagonal (weights), S_ij = w_i * D_ij.

    let n = nodes.len();
    let mut s = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            s[[i, j]] = weights[i] * d[[i, j]];
        }
    }

    Ok(s)
}

/// Compute differentiation matrix D_ij = l'_j(x_i)
pub fn compute_diff_matrix(
    vandermonde: &Array2<f64>,
    nodes: &Array1<f64>,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n = nodes.len();
    let n_modes = vandermonde.ncols();
    let mut vr = Array2::zeros((n, n_modes));

    match basis_type {
        BasisType::Legendre => {
            for i in 0..n {
                let xi = nodes[i];
                for j in 0..n_modes {
                    let (_, p_prime) = legendre_poly_and_deriv(j, xi);
                    let norm_factor = ((2 * j + 1) as f64 / 2.0).sqrt();
                    vr[[i, j]] = p_prime * norm_factor;
                }
            }
        }
        _ => {
            return Err(KwaversError::Numerical(NumericalError::NotImplemented {
                feature: "Differentiation matrix only implemented for Legendre".to_string(),
            }))
        }
    }

    // Compute V^-1
    let v_inv = matrix_inverse(vandermonde)?;

    // D = Vr * V_inv
    let d = vr.dot(&v_inv);

    Ok(d)
}

/// Compute lift matrix
pub fn compute_lift_matrix(
    mass_matrix: &Array2<f64>,
    n_nodes: usize,
) -> KwaversResult<Array2<f64>> {
    let mut e = Array2::zeros((n_nodes, 2));
    e[[0, 0]] = 1.0;
    e[[n_nodes - 1, 1]] = 1.0;

    let mut l = Array2::zeros((n_nodes, 2));
    for i in 0..n_nodes {
        let inv_mass = 1.0 / mass_matrix[[i, i]];
        l[[i, 0]] = inv_mass * e[[i, 0]];
        l[[i, 1]] = inv_mass * e[[i, 1]];
    }

    Ok(l)
}

/// Simple matrix inversion using Gauss-Jordan elimination
pub fn matrix_inverse(a: &Array2<f64>) -> KwaversResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(KwaversError::DimensionMismatch(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    let mut aug = Array2::zeros((n, 2 * n));

    // Initialize augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, i + n]] = 1.0;
    }

    // Gauss-Jordan
    for i in 0..n {
        // Pivot
        let mut pivot = aug[[i, i]];
        let mut pivot_row = i;

        for k in i + 1..n {
            if aug[[k, i]].abs() > pivot.abs() {
                pivot = aug[[k, i]];
                pivot_row = k;
            }
        }

        if pivot.abs() < 1e-10 {
            return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                operation: "matrix_inverse".to_string(),
                condition_number: 0.0,
            }));
        }

        // Swap rows
        if pivot_row != i {
            for j in 0..2 * n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = temp;
            }
        }

        // Scale row
        for j in 0..2 * n {
            aug[[i, j]] /= pivot;
        }

        // Eliminate
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..2 * n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(inv)
}

fn legendre_poly_and_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut l_prev = 1.0;
    let mut l_curr = x;

    for i in 1..n {
        let l_next = ((2 * i + 1) as f64 * x * l_curr - i as f64 * l_prev) / ((i + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
    }

    let deriv = (n as f64) * (l_prev - x * l_curr) / (1.0 - x * x);
    (l_curr, deriv)
}
