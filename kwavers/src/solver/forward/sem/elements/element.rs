//! Hexahedral spectral element.
//!
//! # Isoparametric mapping
//!
//! Each `SemElement` stores the Jacobian fields pre-computed at all
//! `(n_gll)³` GLL quadrature points so that the inner time-step loop
//! requires no further coordinate transformations.
//!
//! The Jacobian `J`, its determinant, and its inverse are computed by
//! [`jacobian::compute_jacobian`] using the trilinear hexahedral shape
//! functions; see that module for the full derivation.
//!
//! # Element volume
//!
//! The volume is evaluated by Gaussian quadrature over the reference
//! element:
//!
//! ```text
//!   V = ∫_{-1}^{1}³ det(J(ξ,η,ζ)) dξ dη dζ
//!     ≈ Σ_{i,j,k} w_i w_j w_k · det(J_{ijk})
//! ```
//!
//! where `w_i` are the GLL weights.  This formula is exact to spectral
//! accuracy for smooth hexahedral elements (Komatitsch & Tromp 1999, §2).
//!
//! # References
//! - Komatitsch & Tromp (1999). GJI 139, §2.
//! - Hughes (2000). *The Finite Element Method*, §3.7.

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2, Array3, Array5};

use super::super::basis::SemBasis;
use super::jacobian::compute_jacobian;

/// Hexahedral spectral element.
///
/// Pre-computes and caches the Jacobian, its determinant, and its inverse
/// at every GLL quadrature point so that downstream assembly routines pay
/// no coordinate-mapping cost per time step.
#[derive(Debug, Clone)]
pub struct SemElement {
    /// Element ID (0-based).
    pub id: usize,
    /// Corner-node physical coordinates.  Shape: `(8, 3)`.
    pub nodes: Array2<f64>,
    /// Jacobian matrix at each GLL point.  Shape: `(n_gll, n_gll, n_gll, 3, 3)`.
    pub jacobian: Array5<f64>,
    /// `det(J)` at each GLL point.  Shape: `(n_gll, n_gll, n_gll)`.
    pub jacobian_det: Array3<f64>,
    /// `J⁻¹` at each GLL point.  Shape: `(n_gll, n_gll, n_gll, 3, 3)`.
    pub jacobian_inv: Array5<f64>,
    /// GLL quadrature weights (length `n_gll`).
    pub gll_weights: Array1<f64>,
}

impl SemElement {
    /// Create a spectral element from 8 corner-node coordinates.
    ///
    /// Validates the node array shape, checks for non-finite coordinates,
    /// then pre-computes and stores the Jacobian fields at all GLL points.
    ///
    /// # Errors
    /// - `InvalidInput` if `nodes` is not `(8, 3)`.
    /// - `InvalidInput` if any node coordinate is NaN or infinite.
    /// - `InvalidInput` if `det(J) ≤ 0` at any GLL point (inverted element).
    /// - `NumericalError::SingularMatrix` if `det(J)` is below the singularity
    ///   threshold `1e-12`.
    pub fn new(id: usize, nodes: Array2<f64>, basis: &SemBasis) -> KwaversResult<Self> {
        if nodes.nrows() != 8 || nodes.ncols() != 3 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "hexahedral element requires 8 nodes with 3 coordinates each, got {}×{}",
                nodes.nrows(),
                nodes.ncols()
            )));
        }

        if !nodes.iter().all(|v| v.is_finite()) {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "hexahedral element nodes contain NaN/Inf".to_owned(),
            ));
        }

        let n_gll = basis.n_points();

        let mut jacobian = Array5::<f64>::zeros((n_gll, n_gll, n_gll, 3, 3));
        let mut jacobian_det = Array3::<f64>::zeros((n_gll, n_gll, n_gll));
        let mut jacobian_inv = Array5::<f64>::zeros((n_gll, n_gll, n_gll, 3, 3));

        for i in 0..n_gll {
            for j in 0..n_gll {
                for k in 0..n_gll {
                    let xi = basis.gll_points[i];
                    let eta = basis.gll_points[j];
                    let zeta = basis.gll_points[k];

                    let (jac, det, inv) = compute_jacobian(&nodes, xi, eta, zeta)?;

                    if !det.is_finite() {
                        return Err(crate::core::error::KwaversError::InvalidInput(
                            "non-finite Jacobian determinant".to_owned(),
                        ));
                    }
                    if det <= 0.0 {
                        return Err(crate::core::error::KwaversError::InvalidInput(format!(
                            "inverted hexahedral element: det(J) = {det:.3e} ≤ 0 at GLL \
                             index ({i},{j},{k})"
                        )));
                    }

                    jacobian
                        .slice_mut(ndarray::s![i, j, k, .., ..])
                        .assign(&jac);
                    jacobian_det[[i, j, k]] = det;
                    jacobian_inv
                        .slice_mut(ndarray::s![i, j, k, .., ..])
                        .assign(&inv);
                }
            }
        }

        Ok(Self {
            id,
            nodes,
            jacobian,
            jacobian_det,
            jacobian_inv,
            gll_weights: basis.gll_weights.clone(),
        })
    }

    /// Map reference coordinates `(ξ, η, ζ) ∈ [-1,1]³` to physical space.
    ///
    /// Uses the same trilinear shape functions as the Jacobian computation.
    #[must_use]
    pub fn reference_to_physical(&self, xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
        let n = [
            (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0,
            (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0,
            (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0,
            (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0,
            (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0,
            (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0,
            (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0,
            (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0,
        ];

        let mut x = [0.0f64; 3];
        for (coord, x_coord) in x.iter_mut().enumerate() {
            for (a, n_a) in n.iter().enumerate() {
                *x_coord += n_a * self.nodes[[a, coord]];
            }
        }
        x
    }

    /// Element volume computed by GLL quadrature.
    ///
    /// ```text
    ///   V = Σ_{i,j,k} w_i w_j w_k · det(J_{ijk})
    /// ```
    #[must_use]
    pub fn volume(&self) -> f64 {
        let n_gll = self.gll_weights.len();
        let mut v = 0.0;
        for i in 0..n_gll {
            let wi = self.gll_weights[i];
            for j in 0..n_gll {
                let wij = wi * self.gll_weights[j];
                for k in 0..n_gll {
                    v += self.jacobian_det[[i, j, k]] * wij * self.gll_weights[k];
                }
            }
        }
        v
    }
}
