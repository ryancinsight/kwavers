//! DG matrix computations: Vandermonde, mass, stiffness, differentiation, and LIFT.
//!
//! ## Algorithm: Mass matrix via GLL quadrature
//!
//! With Gauss-Lobatto-Legendre (GLL) quadrature, the mass matrix is diagonal:
//! ```text
//!   M_{ij} = ∫₋₁¹ φᵢ(x) φⱼ(x) dx ≈ Σ_k w_k φᵢ(x_k) φⱼ(x_k) = w_i δᵢⱼ
//! ```
//! where `w_k` are GLL weights.  Diagonality holds because GLL nodes are the
//! interpolation points, making `φᵢ(x_k) = δᵢₖ` (Lagrange basis property).
//!
//! ## Algorithm: Differentiation matrix `D = Vr · V⁻¹`
//!
//! `Vr_{ij} = φ'_j(xᵢ)` (derivative Vandermonde).  Then:
//! ```text
//!   D_{ij} = (Vr · V⁻¹)_{ij}
//! ```
//! For normalised Legendre: `φ'_j(x) = P̃'_j(x)` computed via the Legendre
//! derivative recurrence. At GLL endpoints the quotient recurrence is replaced
//! by the analytic limit
//! `P'_n(1)=n(n+1)/2`, `P'_n(-1)=(-1)^(n+1)n(n+1)/2`, avoiding the removable
//! singularity at `1 - x² = 0`.
//! For real Fourier modes with `θ = π(x+1)`:
//! ```text
//!   d/dx sin(kθ) = kπ cos(kθ)
//!   d/dx cos(kθ) = -kπ sin(kθ).
//! ```
//!
//! ## Algorithm: Stiffness matrix `S = M · D`
//!
//! ```text
//!   S_{ij} = M_{ii} · D_{ij} = w_i · D_{ij}
//! ```
//!
//! ## Algorithm: LIFT matrix for boundary flux lifting
//!
//! The LIFT matrix maps face residuals `(f* − f)_{face}` to volume DOFs:
//! ```text
//!   LIFT = M⁻¹ · E    where E_{ij} = φᵢ(face node j)
//! ```
//! For 1D: face nodes are `x=−1` (left) and `x=1` (right);
//! `E` is the `n_nodes × 2` face extraction matrix.
//!
//! ## References
//!
//! - Hesthaven & Warburton (2008). *Nodal Discontinuous Galerkin Methods*. Springer. §3.3.
//! - Kopriva (2009). *Implementing Spectral Methods*. Springer. §4.5.

use super::basis::{fourier_theta, validate_fourier_nodes, BasisType};
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, NumericalError};
use kwavers_math::special::legendre::legendre_poly_and_deriv;
use leto::{
    Array1,
    Array2,
};

/// Compute mass matrix using quadrature
/// M_ij = integral(phi_i * phi_j)
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
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
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
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
        BasisType::Chebyshev => {
            for i in 0..n {
                let xi = nodes[i];
                for j in 0..n_modes {
                    vr[[i, j]] = chebyshev_t_derivative(j, xi);
                }
            }
        }
        BasisType::Fourier => {
            validate_fourier_nodes(nodes)?;
            for i in 0..n {
                let theta = fourier_theta(nodes[i]);
                for j in 0..n_modes {
                    // Nyquist mode for even N: the basis column is cos((N/2)θ)
                    // (not the degenerate sin((N/2)θ)), so its derivative is
                    // -(N/2)π sin((N/2)θ).  Must match `fourier_vandermonde_entry`.
                    let is_nyquist = n_modes.is_multiple_of(2) && j == n_modes - 1;
                    vr[[i, j]] = if is_nyquist {
                        let k = (n_modes / 2) as f64;
                        -k * std::f64::consts::PI * (k * theta).sin()
                    } else {
                        real_fourier_basis_derivative(j, theta)
                    };
                }
            }
        }
    }

    // Compute V^-1
    let v_inv = matrix_inverse(vandermonde)?;

    // D = Vr * V_inv
    let d = vr.dot(&v_inv);

    Ok(d)
}

/// Compute lift matrix
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
/// # Errors
/// - Returns [`KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
/// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
///
pub fn matrix_inverse(a: &Array2<f64>) -> KwaversResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(KwaversError::DimensionMismatch(
            "Matrix must be square for inversion".to_owned(),
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
                operation: "matrix_inverse".to_owned(),
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

fn chebyshev_t_derivative(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if x == 1.0 {
        return (n * n) as f64;
    }
    if x == -1.0 {
        let magnitude = (n * n) as f64;
        return if n.is_multiple_of(2) {
            -magnitude
        } else {
            magnitude
        };
    }

    n as f64 * chebyshev_u(n - 1, x)
}

fn chebyshev_u(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }

    let mut u_prev = 1.0;
    let mut u_curr = 2.0 * x;

    for _ in 1..n {
        let u_next = (2.0 * x).mul_add(u_curr, -u_prev);
        u_prev = u_curr;
        u_curr = u_next;
    }

    u_curr
}

fn real_fourier_basis_derivative(mode: usize, theta: f64) -> f64 {
    if mode == 0 {
        return 0.0;
    }

    let wavenumber = mode.div_ceil(2) as f64;
    let angular_wavenumber = wavenumber * std::f64::consts::PI;
    if mode % 2 == 1 {
        angular_wavenumber * (wavenumber * theta).cos()
    } else {
        -angular_wavenumber * (wavenumber * theta).sin()
    }
}

#[cfg(test)]
mod tests;
