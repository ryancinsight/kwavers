//! DG basis functions: Legendre and Chebyshev polynomial families.
//!
//! ## Theorem: Completeness of Legendre polynomials (Hesthaven & Warburton 2008, §3.2)
//!
//! The normalised Legendre polynomials `{P̃_n}_{n=0}^∞` form a complete orthonormal
//! basis for `L²([-1,1])`:
//! ```text
//!   ∫₋₁¹ P̃_i(x) P̃_j(x) dx = δᵢⱼ
//! ```
//! where `P̃_n(x) = sqrt((2n+1)/2) · P_n(x)` and `P_n` is the standard Legendre
//! polynomial.
//!
//! ## Algorithm: Bonnet three-term recurrence (Legendre)
//!
//! ```text
//!   P_0(x) = 1,  P_1(x) = x
//!   P_{n+1}(x) = ((2n+1) x P_n(x) − n P_{n-1}(x)) / (n+1)
//! ```
//!
//! Derivative via Rodrigues' formula (avoiding the singular endpoint form for n ≥ 2):
//! ```text
//!   P'_n(x) = n (P_{n-1}(x) − x P_n(x)) / (1 − x²)   for x ≠ ±1
//!   P'_n(±1) = ±n(n+1)/2                                 (limiting value)
//! ```
//!
//! ## Algorithm: Vandermonde matrix
//!
//! Given collocation nodes `{xᵢ}` and basis functions `{φⱼ}`:
//! ```text
//!   V_{ij} = φⱼ(xᵢ)
//! ```
//! For Legendre DG, `φⱼ = P̃_j` (normalised), so `V_{ij} = P̃_j(xᵢ)`.
//! The inverse `V⁻¹` maps nodal values to modal (expansion) coefficients.
//!
//! ## References
//!
//! - Hesthaven & Warburton (2008). *Nodal Discontinuous Galerkin Methods*. Springer. §3.
//! - Kopriva (2009). *Implementing Spectral Methods*. Springer. §4.

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisType {
    Legendre,
    Chebyshev,
    Fourier,
}

/// Build Vandermonde matrix for given nodes and basis
/// V_ij = phi_j(xi_i)
pub fn build_vandermonde(
    nodes: &Array1<f64>,
    poly_order: usize,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n_nodes = nodes.len();
    let n_modes = poly_order + 1;

    if n_nodes != n_modes {
        // For collocation/nodal DG, usually n_nodes = n_modes
        // But we can build V for oversampling too.
    }

    let mut v = Array2::zeros((n_nodes, n_modes));

    match basis_type {
        BasisType::Legendre => {
            for i in 0..n_nodes {
                let xi = nodes[i];
                for j in 0..n_modes {
                    // Use normalized Legendre polynomials
                    // P_j(x) normalized so that integral_{-1}^1 P_j^2 dx = 1
                    // Standard Legendre: integral = 2/(2j+1)
                    // So multiply by sqrt((2j+1)/2)
                    let p_val = legendre_poly(j, xi);
                    let norm_factor = ((2 * j + 1) as f64 / 2.0).sqrt();
                    v[[i, j]] = p_val * norm_factor;
                }
            }
        }
        _ => {
            return Err(crate::core::error::KwaversError::NotImplemented(format!(
                "Basis type {:?} not implemented",
                basis_type
            )))
        }
    }

    Ok(v)
}

fn legendre_poly(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut l_prev = 1.0;
    let mut l_curr = x;

    for i in 1..n {
        let l_next = ((2 * i + 1) as f64 * x * l_curr - i as f64 * l_prev) / ((i + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}
