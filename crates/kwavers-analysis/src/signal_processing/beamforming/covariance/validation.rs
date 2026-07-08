use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

/// Validate covariance matrix structure: square, Hermitian, finite values.
///
/// Checks:
/// 1. Square (N×N)
/// 2. All entries finite
/// 3. Hermitian within tolerance 1e-10
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn validate_covariance_matrix(covariance: &Array2<Complex64>) -> KwaversResult<()> {
    let (nrows, ncols) = (covariance.nrows(), covariance.ncols());

    if nrows != ncols {
        return Err(KwaversError::InvalidInput(format!(
            "Covariance matrix must be square, got shape ({}, {})",
            nrows, ncols
        )));
    }

    if !covariance
        .iter()
        .all(|&x| x.re.is_finite() && x.im.is_finite())
    {
        return Err(KwaversError::InvalidInput(
            "Covariance matrix contains non-finite values (NaN or Inf)".into(),
        ));
    }

    if !is_hermitian(covariance, 1e-10) {
        return Err(KwaversError::InvalidInput(
            "Covariance matrix is not Hermitian".into(),
        ));
    }

    Ok(())
}

/// Check if a matrix is Hermitian within numerical tolerance.
///
/// Returns `true` if ||A − A^H||_∞ ≤ tolerance.
/// A is Hermitian iff `A[i,j] = A[j,i]^*` for all i,j.
#[must_use]
pub fn is_hermitian(matrix: &Array2<Complex64>, tolerance: f64) -> bool {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return false;
    }

    for i in 0..n {
        if matrix[[i, i]].im.abs() > tolerance {
            return false;
        }

        for j in (i + 1)..n {
            let diff = matrix[[i, j]] - matrix[[j, i]].conj();
            if diff.norm() > tolerance {
                return false;
            }
        }
    }

    true
}

/// Compute the trace of a square matrix: `tr(A) = ∑ᵢ A[i,i]`.
///
/// For covariance matrices, the trace equals total signal power across sensors.
///
/// # Errors
///
/// Returns `Err` if the matrix is not square.
pub fn trace(matrix: &Array2<Complex64>) -> KwaversResult<Complex64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(KwaversError::InvalidInput(format!(
            "Trace requires square matrix, got shape ({}, {})",
            matrix.nrows(),
            matrix.ncols()
        )));
    }

    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..matrix.nrows() {
        sum += matrix[[i, i]];
    }

    Ok(sum)
}
