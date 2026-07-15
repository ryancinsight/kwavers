use crate::linear_algebra::eigendecomposition::{EigenSolver, EigenSolverConfig};
use crate::linear_algebra::{ComplexLinearAlgebra, VectorOperations};
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};

/// Compute L2 norm of a 3D array.
#[must_use]
pub fn norm_l2(array: &leto::Array3<f64>) -> f64 {
    VectorOperations::norm_l2(array)
}

/// Extension trait providing fluent leto linear-algebra operations.
pub trait LinearAlgebraExt<T> {
    /// Solve linear system `self x = b`.
    /// # Errors
    /// - Propagates any `KwaversError` from the underlying solver.
    fn solve_into(&self, b: Array1<T>) -> KwaversResult<Array1<T>>;

    /// Compute matrix inverse.
    /// # Errors
    /// - Propagates any `KwaversError` from the underlying solver.
    fn inv(&self) -> KwaversResult<Array2<T>>;

    /// Eigendecomposition.
    /// # Errors
    /// - Propagates any `KwaversError` from the underlying solver.
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
        // The Hermitian eigensolver operates on complex matrices; a real matrix
        // is promoted with zero imaginary part. For symmetric real inputs the
        // eigenvectors are real (imaginary part ~0), so projecting to the real
        // part preserves the original real-valued contract of this method.
        let n = self.shape()[0];
        let mut complex = Array2::<Complex64>::zeros([n, n]);
        for i in 0..n {
            for j in 0..n {
                complex[[i, j]] = Complex64::new(self[[i, j]], 0.0);
            }
        }
        let result = EigenSolver::qr_algorithm(&complex, EigenSolverConfig::default())?;
        let mut eigenvectors = Array2::<f64>::zeros([n, n]);
        for i in 0..n {
            for j in 0..n {
                eigenvectors[[i, j]] = result.eigenvectors[[i, j]].re;
            }
        }
        Ok((result.eigenvalues, eigenvectors))
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
        let result = EigenSolver::jacobi_hermitian(self, EigenSolverConfig::default())?;
        let n = result.eigenvalues.shape()[0];
        let mut eig_complex = Array1::<Complex64>::zeros([n]);
        for i in 0..n {
            eig_complex[i] = Complex64::new(result.eigenvalues[i], 0.0);
        }
        Ok((eig_complex, result.eigenvectors))
    }
}
