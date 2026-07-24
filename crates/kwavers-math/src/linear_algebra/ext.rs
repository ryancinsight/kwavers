//! Extension trait providing fluent leto linear-algebra operations.
//!
//! SSOT: leto-ops. This trait is a thin kwavers-vocabulary wrapper that
//! delegates to leto-ops functions, preserving the KwaversError contract.

use crate::linear_algebra::complex::ComplexLinearAlgebra;
use crate::linear_algebra::tolerance;
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};

/// Compute L2 norm of a 3D array (delegates to leto_ops::norm_l2).
#[must_use]
pub fn norm_l2(array: &leto::Array3<f64>) -> f64 {
    leto_ops::norm_l2(&array.view()).expect("kwavers arrays must have valid leto layouts")
}

/// Extension trait providing fluent leto linear-algebra operations.
pub trait LinearAlgebraExt<T> {
    /// Solve linear system self x = b.
    /// # Errors
    /// - Propagates any KwaversError from the underlying solver.
    fn solve_into(&self, b: Array1<T>) -> KwaversResult<Array1<T>>;

    /// Compute matrix inverse.
    /// # Errors
    /// - Propagates any KwaversError from the underlying solver.
    fn inv(&self) -> KwaversResult<Array2<T>>;

    /// Eigendecomposition.
    /// # Errors
    /// - Propagates any KwaversError from the underlying solver.
    fn eig(&self) -> KwaversResult<(Array1<T>, Array2<T>)>;
}

impl LinearAlgebraExt<f64> for Array2<f64> {
    fn solve_into(&self, b: Array1<f64>) -> KwaversResult<Array1<f64>> {
        Ok(leto_ops::solve(&self.view(), &b.view())?)
    }

    fn inv(&self) -> KwaversResult<Self> {
        Ok(leto_ops::inv(&self.view())?)
    }

    fn eig(&self) -> KwaversResult<(Array1<f64>, Self)> {
        // Delegate to leto-ops symmetric eigensolver (Jacobi).
        let result = leto_ops::symmetric_eigen_jacobi(&self.view())?;
        Ok((
            Array1::from_vec([result.eigenvalues.len()], result.eigenvalues)
                .expect("eigenvalue vector length must match"),
            result.eigenvectors,
        ))
    }
}

impl LinearAlgebraExt<Complex64> for Array2<Complex64> {
    fn solve_into(&self, b: Array1<Complex64>) -> KwaversResult<Array1<Complex64>> {
        ComplexLinearAlgebra::solve_linear_system_complex(self, &b)
    }

    fn inv(&self) -> KwaversResult<Self> {
        ComplexLinearAlgebra::matrix_inverse_complex(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<Complex64>, Self)> {
        // Delegate to leto-ops Hermitian eigensolver (Jacobi).
        let config = leto_ops::HermitianEigenConfig {
            tolerance: tolerance::HERMITIAN_EIG_TOL,
            max_iterations: tolerance::HERMITIAN_EIG_MAX_SWEEPS,
            sort_descending: true,
            estimate_condition: true,
        };
        let result = leto_ops::hermitian_eigen_jacobi(self, config)?;
        Ok((
            result.eigenvalues.mapv(|value| Complex64::new(value, 0.0)),
            result.eigenvectors,
        ))
    }
}
