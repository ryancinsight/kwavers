//! Frangi multi-scale vesselness filter.
//!
//! # Mathematical specification
//!
//! ## Hessian matrix
//!
//! At voxel `(i,j,k)` the second-order derivatives are computed by the standard
//! central-difference stencils:
//!
//! ```text
//!   H_xx = I[i+1,j,k] − 2I[i,j,k] + I[i−1,j,k]
//!   H_xy = (I[i+1,j+1,k] − I[i−1,j+1,k] − I[i+1,j−1,k] + I[i−1,j−1,k]) / 4
//!   ⋮ (similarly for all 6 independent entries)
//! ```
//!
//! ## Eigenvalues
//!
//! Analytical eigenvalues of the 3×3 symmetric Hessian are computed via
//! Cardano's method (Smith 1961; Deledalle et al. 2010):
//!
//! 1. Shift: `q = trace(H) / 3`.
//! 2. Scale: `p = ‖H − qI‖_F / √6`.
//! 3. Phase: `φ = arccos(det(B)/2) / 3`  where `B = (H − qI) / p`.
//! 4. Eigenvalues: `λ₁ = q + 2p cos φ`,  `λ₃ = q + 2p cos(φ + 2π/3)`,
//!    `λ₂ = 3q − λ₁ − λ₃`.
//!
//! ## Vesselness measure (Frangi et al. 1998)
//!
//! Sort `|λ₁| ≤ |λ₂| ≤ |λ₃|`.  A bright tubular structure on dark background
//! has `|λ₂| ≈ |λ₃| ≫ |λ₁|` with `λ₂, λ₃ < 0`.
//!
//! ```text
//!   R_A = |λ₂| / |λ₃|         (plate vs. line)
//!   R_B = |λ₁| / √(|λ₂ λ₃|)  (blob vs. tube)
//!   S   = √(λ₁² + λ₂² + λ₃²) (structure magnitude)
//!
//!   V = (1 − exp(−R_A²/2α²)) · exp(−R_B²/2β²) · (1 − exp(−S²/2c²))
//! ```
//!
//! where `α = β = 0.5` (standard defaults) and `c = s_max / 2` (adaptive).
//!
//! # References
//! - Frangi et al. (1998). MICCAI 1496, pp. 130-137.
//! - Smith (1961). J. ACM 8(1), pp. 105-111.

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Compute the Frangi vesselness response for a 3-D image.
///
/// Boundary voxels (within 1 voxel of any edge) are left at zero because the
/// central-difference stencil requires all 6-neighbours.
///
/// # Errors
/// Never errors in practice; the `KwaversResult` signature accommodates future
/// multi-scale extensions that may fail on memory allocation.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(super) fn compute_frangi_response(image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = image.dim();
    let mut response = Array3::zeros((nx, ny, nz));

    // Standard Frangi parameters.
    let alpha = 0.5_f64;
    let beta = 0.5_f64;
    let two_alpha_sq = 2.0 * alpha * alpha;
    let two_beta_sq = 2.0 * beta * beta;

    // First pass: find the maximum structure magnitude for the adaptive `c`.
    let mut s_max = 0.0_f64;
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let h = hessian_at(image, i, j, k);
                let (l1, l2, l3) = symmetric_3x3_eigenvalues(h);
                let s = l3.mul_add(l3, l1.mul_add(l1, l2 * l2)).sqrt();
                if s > s_max {
                    s_max = s;
                }
            }
        }
    }
    let c = s_max.mul_add(0.5, 1e-30);
    let two_c_sq = 2.0 * c * c;

    // Second pass: compute vesselness at each interior voxel.
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let h = hessian_at(image, i, j, k);
                let (e1, e2, e3) = symmetric_3x3_eigenvalues(h);

                // Sort by absolute value so |a1| ≤ |a2| ≤ |a3|.
                let mut sorted = [e1.abs(), e2.abs(), e3.abs()];
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let (a1, a2, a3) = (sorted[0], sorted[1], sorted[2]);

                if a3 < 1e-30 {
                    continue;
                }

                let r_a = a2 / a3;
                let r_b = a1 / (a2 * a3).sqrt().max(1e-30);
                let s = a3.mul_add(a3, a1.mul_add(a1, a2 * a2)).sqrt();

                response[[i, j, k]] = (1.0 - (-r_a * r_a / two_alpha_sq).exp())
                    * (-r_b * r_b / two_beta_sq).exp()
                    * (1.0 - (-s * s / two_c_sq).exp());
            }
        }
    }

    Ok(response)
}

/// Compute the 6 independent Hessian components at interior voxel `(i,j,k)`.
///
/// Returns `[H_xx, H_yy, H_zz, H_xy, H_xz, H_yz]`.
pub(super) fn hessian_at(image: &Array3<f64>, i: usize, j: usize, k: usize) -> [f64; 6] {
    let c = image[[i, j, k]];
    let hxx = 2.0f64.mul_add(-c, image[[i + 1, j, k]]) + image[[i - 1, j, k]];
    let hyy = 2.0f64.mul_add(-c, image[[i, j + 1, k]]) + image[[i, j - 1, k]];
    let hzz = 2.0f64.mul_add(-c, image[[i, j, k + 1]]) + image[[i, j, k - 1]];
    let hxy = (image[[i + 1, j + 1, k]] - image[[i - 1, j + 1, k]] - image[[i + 1, j - 1, k]]
        + image[[i - 1, j - 1, k]])
        / 4.0;
    let hxz = (image[[i + 1, j, k + 1]] - image[[i - 1, j, k + 1]] - image[[i + 1, j, k - 1]]
        + image[[i - 1, j, k - 1]])
        / 4.0;
    let hyz = (image[[i, j + 1, k + 1]] - image[[i, j - 1, k + 1]] - image[[i, j + 1, k - 1]]
        + image[[i, j - 1, k - 1]])
        / 4.0;
    [hxx, hyy, hzz, hxy, hxz, hyz]
}

/// Analytical eigenvalues of a 3×3 symmetric matrix via Cardano's method.
///
/// # Input format
/// `h = [a11, a22, a33, a12, a13, a23]`
///
/// # Algorithm
/// Uses the real-symmetric Cardano path (Smith 1961).  The `det(B)/2` value
/// is clamped to `[-1, 1]` before `arccos` to guard against floating-point
/// rounding outside the domain.
pub(super) fn symmetric_3x3_eigenvalues(h: [f64; 6]) -> (f64, f64, f64) {
    let (a11, a22, a33) = (h[0], h[1], h[2]);
    let (a12, a13, a23) = (h[3], h[4], h[5]);

    let q = (a11 + a22 + a33) / 3.0;
    let p1 = a23.mul_add(a23, a12.mul_add(a12, a13 * a13));

    if p1 < 1e-30 {
        return (a11, a22, a33); // already diagonal
    }

    let p2 = 2.0f64.mul_add(p1, (a33 - q).mul_add(a33 - q, (a22 - q).mul_add(a22 - q, (a11 - q).powi(2))));
    let p = (p2 / 6.0).sqrt();
    let inv_p = 1.0 / p;

    // B = (A − qI) / p
    let b11 = (a11 - q) * inv_p;
    let b22 = (a22 - q) * inv_p;
    let b33 = (a33 - q) * inv_p;
    let b12 = a12 * inv_p;
    let b13 = a13 * inv_p;
    let b23 = a23 * inv_p;

    let det_b = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13)
        + b13 * (b12 * b23 - b22 * b13);

    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let eig1 = (2.0 * p).mul_add(phi.cos(), q);
    let eig3 = (2.0 * p).mul_add((phi + std::f64::consts::TAU / 3.0).cos(), q);
    let eig2 = 3.0f64.mul_add(q, -eig1) - eig3;

    (eig1, eig2, eig3)
}
