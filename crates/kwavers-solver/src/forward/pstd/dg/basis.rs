//! DG basis functions: Legendre, Chebyshev, and real Fourier families.
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
//!   P'_n(1) = n(n+1)/2,  P'_n(-1) = (-1)^(n+1)n(n+1)/2    (limiting values)
//! ```
//!
//! ## Algorithm: Chebyshev first-kind recurrence
//!
//! ```text
//!   T_0(x) = 1,  T_1(x) = x
//!   T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
//! ```
//!
//! The derivative satisfies `T'_n(x) = n U_{n-1}(x)`, where `U_m` is the
//! Chebyshev polynomial of the second kind. Endpoint limits are
//! `T'_n(1)=n²` and `T'_n(-1)=(-1)^(n-1)n²`.
//!
//! ## Algorithm: real Fourier basis on `[-1, 1)`
//!
//! For periodic reference coordinate `x`, with `θ = π(x + 1)`, the real basis is
//! ```text
//!   φ₀(x) = 1
//!   φ₂k₋₁(x) = sin(kθ),  φ₂k(x) = cos(kθ),  k ≥ 1.
//! ```
//! This spans the real trigonometric polynomial space through wavenumber
//! `ceil(p/2)`. The basis is valid only on periodic node sets that do not
//! contain both `x=-1` and `x=1`, because those coordinates represent the same
//! point on the reference torus and make interpolation matrices singular.
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

use kwavers_core::error::KwaversResult;
use kwavers_math::special::legendre::legendre_poly;
use leto::{Array1, Array2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisType {
    Legendre,
    Chebyshev,
    Fourier,
}

/// Build Vandermonde matrix for given nodes and basis
/// V_ij = phi_j(xi_i)
/// # Errors
/// - Propagates any [`crate::KwaversError`] returned by called functions.
///
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
        BasisType::Chebyshev => {
            for i in 0..n_nodes {
                let xi = nodes[i];
                for j in 0..n_modes {
                    v[[i, j]] = chebyshev_t(j, xi);
                }
            }
        }
        BasisType::Fourier => {
            validate_fourier_nodes(nodes)?;
            for i in 0..n_nodes {
                let theta = fourier_theta(nodes[i]);
                for j in 0..n_modes {
                    // For even N=2M the last mode (j=N-1, which is odd) would
                    // normally be sin(Mθ), but sin(Mθ) = sin(jπ) = 0 at every
                    // equispaced node x_j = -1+2j/N (discrete Nyquist degeneracy).
                    // Replace it with the non-degenerate Nyquist cosine cos(Mθ),
                    // which completes the real Fourier basis and is invertible.
                    // Reference: Brigham (1988) §4.3; Hesthaven & Warburton (2008) §5.3.
                    v[[i, j]] = fourier_vandermonde_entry(n_modes, j, theta);
                }
            }
        }
    }

    Ok(v)
}

pub(super) fn chebyshev_t(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut t_prev = 1.0;
    let mut t_curr = x;

    for _ in 1..n {
        let t_next = (2.0 * x).mul_add(t_curr, -t_prev);
        t_prev = t_curr;
        t_curr = t_next;
    }

    t_curr
}
/// Validate fourier nodes.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn validate_fourier_nodes(nodes: &Array1<f64>) -> KwaversResult<()> {
    const PERIODIC_ENDPOINT_TOL: f64 = 1e-12;

    if nodes.iter().any(|node| !node.is_finite()) {
        return Err(kwavers_core::error::KwaversError::InvalidInput(
            "Fourier basis nodes must be finite".to_owned(),
        ));
    }

    let has_left_endpoint = nodes
        .iter()
        .any(|node| (*node + 1.0).abs() <= PERIODIC_ENDPOINT_TOL);
    let has_right_endpoint = nodes
        .iter()
        .any(|node| (*node - 1.0).abs() <= PERIODIC_ENDPOINT_TOL);

    if has_left_endpoint && has_right_endpoint {
        return Err(kwavers_core::error::KwaversError::InvalidInput(
            "Fourier basis on [-1,1) cannot include both periodic endpoints -1 and 1".to_owned(),
        ));
    }

    Ok(())
}

#[inline]
pub(super) fn fourier_theta(x: f64) -> f64 {
    std::f64::consts::PI * (x + 1.0)
}

/// Evaluate the real Fourier basis function `φ_mode` at angle `theta = π(x+1)`.
///
/// The basis ordering is:
/// - mode 0: 1 (constant)
/// - mode 2k-1: sin(kθ), k ≥ 1
/// - mode 2k:   cos(kθ), k ≥ 1
pub(super) fn real_fourier_basis(mode: usize, theta: f64) -> f64 {
    if mode == 0 {
        return 1.0;
    }

    let wavenumber = mode.div_ceil(2) as f64;
    if mode % 2 == 1 {
        (wavenumber * theta).sin()
    } else {
        (wavenumber * theta).cos()
    }
}

/// Evaluate the real Fourier Vandermonde entry for basis size `n_modes`.
///
/// Handles the discrete Nyquist degeneracy: for even `n_modes = 2M`, mode `N-1`
/// (which is odd) would normally map to `sin(Mθ)`, but `sin(Mθ) = 0` at every
/// equispaced node `x_j = -1 + 2j/N`.  The non-degenerate replacement is the
/// Nyquist cosine `cos(Mθ)`, which keeps the Vandermonde invertible.
#[inline]
pub(super) fn fourier_vandermonde_entry(n_modes: usize, mode: usize, theta: f64) -> f64 {
    let is_nyquist = n_modes.is_multiple_of(2) && mode == n_modes - 1;
    if is_nyquist {
        let k = (n_modes / 2) as f64;
        (k * theta).cos()
    } else {
        real_fourier_basis(mode, theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourier_vandermonde_evaluates_real_trigonometric_basis() {
        let nodes = Array1::from_vec(3, vec![-0.5, 0.0, 0.5]).unwrap();

        let v = build_vandermonde(&nodes, 2, BasisType::Fourier).unwrap();

        assert_eq!(v.shape(), [3, 3]);
        for row in 0..3 {
            assert_eq!(v[[row, 0]], 1.0);
        }
        assert!((v[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!(v[[1, 1]].abs() <= 1e-12);
        assert!((v[[2, 1]] + 1.0).abs() <= 1e-12);
        assert!(v[[0, 2]].abs() <= 1e-12);
        assert!((v[[1, 2]] + 1.0).abs() <= 1e-12);
        assert!(v[[2, 2]].abs() <= 1e-12);
    }

    #[test]
    fn fourier_vandermonde_rejects_duplicate_periodic_endpoints() {
        let nodes = Array1::from_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();

        let error = build_vandermonde(&nodes, 2, BasisType::Fourier).unwrap_err();

        assert!(format!("{error}").contains("cannot include both periodic endpoints"));
    }
}
