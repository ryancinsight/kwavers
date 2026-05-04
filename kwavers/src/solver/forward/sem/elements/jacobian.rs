//! Hexahedral Jacobian computation.
//!
//! # Mathematical specification
//!
//! For a trilinear hexahedral element with 8 corner nodes
//! `{x_a}_{a=1}^{8} ⊂ ℝ³` the isoparametric map from reference
//! coordinates `(ξ, η, ζ) ∈ [-1,1]³` to physical space is
//!
//! ```text
//!   x(ξ,η,ζ) = Σ_{a=1}^{8}  N_a(ξ,η,ζ) · x_a
//! ```
//!
//! where the standard trilinear shape functions are
//!
//! ```text
//!   N_a(ξ,η,ζ) = (1 ± ξ)(1 ± η)(1 ± ζ) / 8
//! ```
//!
//! (signs selected by the local node numbering convention used here).
//!
//! ## Jacobian matrix
//!
//! ```text
//!   J_{αi} = ∂x_α / ∂r_i   where r = (ξ,η,ζ)
//!
//!          = Σ_a (∂N_a / ∂r_i) · x_{a,α}
//! ```
//!
//! The matrix is 3×3.  Its determinant `det(J)` equals the volume
//! scaling factor between reference and physical space; it must be
//! strictly positive for a non-inverted element.
//!
//! ## Inverse Jacobian
//!
//! The analytic cofactor formula (Cramer's rule) is used directly to
//! avoid a linear-system solve.  This is exact for 3×3 and avoids any
//! accumulated round-off from Gaussian elimination.
//!
//! ## Singularity guard
//!
//! `|det(J)| < 1e-12` is treated as a singular (degenerate) element and
//! returns `NumericalError::SingularMatrix`.
//!
//! # References
//! - Hughes (2000). *The Finite Element Method*, §3.7.
//! - Komatitsch & Tromp (1999). GJI 139, §2.

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::Array2;

/// Compute the 3×3 Jacobian matrix, its determinant, and its inverse at
/// reference point `(xi, eta, zeta)` for a trilinear hexahedral element.
///
/// # Arguments
/// * `nodes` — shape `(8, 3)` array of physical corner-node coordinates.
/// * `xi`, `eta`, `zeta` — reference-space coordinates in `[-1, 1]`.
///
/// # Returns
/// `(J, det(J), J⁻¹)` where `J` and `J⁻¹` have shape `(3, 3)`.
///
/// # Errors
/// Returns [`NumericalError::SingularMatrix`] when `|det(J)| < 1e-12`.
pub(super) fn compute_jacobian(
    nodes: &Array2<f64>,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> KwaversResult<(Array2<f64>, f64, Array2<f64>)> {
    // ── Derivatives of shape functions w.r.t. reference coordinates ──────
    // Node ordering:
    //   0:(−,−,−)  1:(+,−,−)  2:(+,+,−)  3:(−,+,−)
    //   4:(−,−,+)  5:(+,−,+)  6:(+,+,+)  7:(−,+,+)
    let dn_dxi: [f64; 8] = [
        -(1.0 - eta) * (1.0 - zeta) / 8.0, // dN1/dξ
        (1.0 - eta) * (1.0 - zeta) / 8.0,  // dN2/dξ
        (1.0 + eta) * (1.0 - zeta) / 8.0,  // dN3/dξ
        -(1.0 + eta) * (1.0 - zeta) / 8.0, // dN4/dξ
        -(1.0 - eta) * (1.0 + zeta) / 8.0, // dN5/dξ
        (1.0 - eta) * (1.0 + zeta) / 8.0,  // dN6/dξ
        (1.0 + eta) * (1.0 + zeta) / 8.0,  // dN7/dξ
        -(1.0 + eta) * (1.0 + zeta) / 8.0, // dN8/dξ
    ];

    let dn_deta: [f64; 8] = [
        -(1.0 - xi) * (1.0 - zeta) / 8.0, // dN1/dη
        -(1.0 + xi) * (1.0 - zeta) / 8.0, // dN2/dη
        (1.0 + xi) * (1.0 - zeta) / 8.0,  // dN3/dη
        (1.0 - xi) * (1.0 - zeta) / 8.0,  // dN4/dη
        -(1.0 - xi) * (1.0 + zeta) / 8.0, // dN5/dη
        -(1.0 + xi) * (1.0 + zeta) / 8.0, // dN6/dη
        (1.0 + xi) * (1.0 + zeta) / 8.0,  // dN7/dη
        (1.0 - xi) * (1.0 + zeta) / 8.0,  // dN8/dη
    ];

    let dn_dzeta: [f64; 8] = [
        -(1.0 - xi) * (1.0 - eta) / 8.0, // dN1/dζ
        -(1.0 + xi) * (1.0 - eta) / 8.0, // dN2/dζ
        -(1.0 + xi) * (1.0 + eta) / 8.0, // dN3/dζ
        -(1.0 - xi) * (1.0 + eta) / 8.0, // dN4/dζ
        (1.0 - xi) * (1.0 - eta) / 8.0,  // dN5/dζ
        (1.0 + xi) * (1.0 - eta) / 8.0,  // dN6/dζ
        (1.0 + xi) * (1.0 + eta) / 8.0,  // dN7/dζ
        (1.0 - xi) * (1.0 + eta) / 8.0,  // dN8/dζ
    ];

    // ── Assemble Jacobian J_{αi} = Σ_a (∂N_a/∂r_i) · x_{a,α} ───────────
    // Index convention: J[[α, i]]  where α ∈ {x,y,z}, i ∈ {ξ,η,ζ}.
    let mut j = Array2::<f64>::zeros((3, 3));
    for alpha in 0..3 {
        for n in 0..8 {
            j[[alpha, 0]] += dn_dxi[n] * nodes[[n, alpha]];
            j[[alpha, 1]] += dn_deta[n] * nodes[[n, alpha]];
            j[[alpha, 2]] += dn_dzeta[n] * nodes[[n, alpha]];
        }
    }

    // ── Determinant (Sarrus / cofactor expansion along row 0) ─────────────
    let det = j[[0, 0]] * (j[[1, 1]] * j[[2, 2]] - j[[1, 2]] * j[[2, 1]])
        - j[[0, 1]] * (j[[1, 0]] * j[[2, 2]] - j[[1, 2]] * j[[2, 0]])
        + j[[0, 2]] * (j[[1, 0]] * j[[2, 1]] - j[[1, 1]] * j[[2, 0]]);

    if det.abs() < 1e-12 {
        return Err(NumericalError::SingularMatrix {
            operation: "SEM Jacobian computation".to_string(),
            condition_number: det.abs(),
        }
        .into());
    }

    // ── Inverse via Cramer's rule: J⁻¹ = adj(J)ᵀ / det(J) ───────────────
    let mut j_inv = Array2::<f64>::zeros((3, 3));

    j_inv[[0, 0]] = (j[[1, 1]] * j[[2, 2]] - j[[1, 2]] * j[[2, 1]]) / det;
    j_inv[[0, 1]] = (j[[0, 2]] * j[[2, 1]] - j[[0, 1]] * j[[2, 2]]) / det;
    j_inv[[0, 2]] = (j[[0, 1]] * j[[1, 2]] - j[[0, 2]] * j[[1, 1]]) / det;

    j_inv[[1, 0]] = (j[[1, 2]] * j[[2, 0]] - j[[1, 0]] * j[[2, 2]]) / det;
    j_inv[[1, 1]] = (j[[0, 0]] * j[[2, 2]] - j[[0, 2]] * j[[2, 0]]) / det;
    j_inv[[1, 2]] = (j[[0, 2]] * j[[1, 0]] - j[[0, 0]] * j[[1, 2]]) / det;

    j_inv[[2, 0]] = (j[[1, 0]] * j[[2, 1]] - j[[1, 1]] * j[[2, 0]]) / det;
    j_inv[[2, 1]] = (j[[0, 1]] * j[[2, 0]] - j[[0, 0]] * j[[2, 1]]) / det;
    j_inv[[2, 2]] = (j[[0, 0]] * j[[1, 1]] - j[[0, 1]] * j[[1, 0]]) / det;

    Ok((j, det, j_inv))
}
