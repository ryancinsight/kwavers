use kwavers_core::error::{KwaversError, KwaversResult};

/// Compute `det(A)` for a 3×3 row-major matrix.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[inline]
fn det3(a: &[f64; 9]) -> f64 {
    a[2].mul_add(
        a[3].mul_add(a[7], -(a[4] * a[6])),
        a[0].mul_add(
            a[4].mul_add(a[8], -(a[5] * a[7])),
            -(a[1] * a[3].mul_add(a[8], -(a[5] * a[6]))),
        ),
    )
}

/// Invert a 3×3 symmetric positive-definite matrix using Cramer's rule.
///
/// Returns `Err` when `|det(A)| < 1e-30` (degenerate).
/// # Errors
/// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
///
pub(super) fn invert3(a: &[f64; 9]) -> KwaversResult<[f64; 9]> {
    let d = det3(a);
    if d.abs() < 1e-30 {
        return Err(KwaversError::InvalidInput(
            "Innovation covariance matrix is singular — cannot invert 3×3 S".to_owned(),
        ));
    }
    let inv_d = 1.0 / d;
    Ok([
        a[4].mul_add(a[8], -(a[5] * a[7])) * inv_d,
        a[2].mul_add(a[7], -(a[1] * a[8])) * inv_d,
        a[1].mul_add(a[5], -(a[2] * a[4])) * inv_d,
        a[5].mul_add(a[6], -(a[3] * a[8])) * inv_d,
        a[0].mul_add(a[8], -(a[2] * a[6])) * inv_d,
        a[2].mul_add(a[3], -(a[0] * a[5])) * inv_d,
        a[3].mul_add(a[7], -(a[4] * a[6])) * inv_d,
        a[1].mul_add(a[6], -(a[0] * a[7])) * inv_d,
        a[0].mul_add(a[4], -(a[1] * a[3])) * inv_d,
    ])
}

/// Compute `C = A·B` for A ∈ ℝ^(6×3) and B ∈ ℝ^(3×3); result ∈ ℝ^(6×3).
pub(super) fn mat6x3_mul_mat3x3(a: &[f64; 18], b: &[f64; 9]) -> [f64; 18] {
    let mut c = [0.0_f64; 18];
    for i in 0..6 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            c[i * 3 + j] = s;
        }
    }
    c
}

/// Compute `C = A·B` for A ∈ ℝ^(6×6) and B ∈ ℝ^(6×6); result ∈ ℝ^(6×6).
pub(super) fn mat6x6_mul_mat6x6(a: &[f64; 36], b: &[f64; 36]) -> [f64; 36] {
    let mut c = [0.0_f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i * 6 + k] * b[k * 6 + j];
            }
            c[i * 6 + j] = s;
        }
    }
    c
}