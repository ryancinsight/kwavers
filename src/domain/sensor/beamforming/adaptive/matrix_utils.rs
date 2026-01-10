//! Matrix utilities for beamforming algorithms
//!
//! # SSOT invariant
//! This module must not implement ad-hoc linear algebra (matrix inversion, eigensolvers, etc.).
//! Those operations are owned by the SSOT `crate::utils::linear_algebra` module.
//!
//! This file therefore provides only compatibility shims so higher-level adaptive algorithms
//! do not duplicate numerics.

use crate::domain::math::linear_algebra::LinearAlgebra;
use ndarray::Array2;
use num_complex::{Complex, Complex64};

/// Matrix inverse (compatibility shim).
///
/// # SSOT
/// This routes to `crate::utils::linear_algebra::LinearAlgebra::matrix_inverse_complex`.
///
/// # Error policy
/// Returns `None` on numerical errors to preserve existing call sites. New SSOT-first
/// code should call the SSOT function directly and propagate `KwaversResult`.
#[must_use]
pub fn invert_matrix(mat: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let a: Array2<Complex<f64>> = mat.mapv(|z| Complex::new(z.re, z.im));
    let inv = LinearAlgebra::matrix_inverse_complex(&a).ok()?;
    Some(inv.mapv(|z| Complex64::new(z.re, z.im)))
}

/// Complex Hermitian eigendecomposition is not provided from this module.
///
/// # SSOT invariant
/// SSOT `LinearAlgebra` currently does not implement complex eigendecomposition. Until SSOT gains a
/// formally verified complex Hermitian eigen-solver, this function returns `None` and callers must
/// be refactored to avoid relying on this.
///
#[must_use]
pub fn eigen_hermitian(
    _mat: &Array2<Complex64>,
    _num_eigs: usize,
) -> Option<(Vec<f64>, Array2<Complex64>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matrix_inversion_identity() {
        let mut mat = Array2::<Complex64>::zeros((3, 3));
        mat[(0, 0)] = Complex64::new(1.0, 0.0);
        mat[(1, 1)] = Complex64::new(1.0, 0.0);
        mat[(2, 2)] = Complex64::new(1.0, 0.0);

        let inv = invert_matrix(&mat).unwrap();
        assert_relative_eq!(inv[(0, 0)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inv[(1, 1)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inv[(2, 2)].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_inversion_singular() {
        let mut mat = Array2::<Complex64>::zeros((2, 2));
        mat[(0, 0)] = Complex64::new(1.0, 0.0);
        mat[(0, 1)] = Complex64::new(2.0, 0.0);
        mat[(1, 0)] = Complex64::new(2.0, 0.0); // Linearly dependent rows
        mat[(1, 1)] = Complex64::new(4.0, 0.0);

        assert!(invert_matrix(&mat).is_none());
    }
}
