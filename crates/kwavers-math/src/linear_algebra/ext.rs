use kwavers_core::error::KwaversResult;
use crate::linear_algebra::{
    ComplexLinearAlgebra, EigenDecomposition, LinearAlgebra, VectorOperations,
};
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Compute L2 norm of a 3D array.
#[must_use]
pub fn norm_l2(array: &ndarray::Array3<f64>) -> f64 {
    VectorOperations::norm_l2(array)
}

/// Extension trait providing fluent ndarray linear-algebra operations.
pub trait LinearAlgebraExt<T> {
    /// Solve linear system `self x = b`.
    /// # Errors
    /// - Propagates any [`KwaversError`] from the underlying solver.
    fn solve_into(&self, b: Array1<T>) -> KwaversResult<Array1<T>>;

    /// Compute matrix inverse.
    /// # Errors
    /// - Propagates any [`KwaversError`] from the underlying solver.
    fn inv(&self) -> KwaversResult<Array2<T>>;

    /// Eigendecomposition.
    /// # Errors
    /// - Propagates any [`KwaversError`] from the underlying solver.
    fn eig(&self) -> KwaversResult<(Array1<T>, Array2<T>)>;
}

impl LinearAlgebraExt<f64> for Array2<f64> {
    fn solve_into(&self, b: Array1<f64>) -> KwaversResult<Array1<f64>> {
        LinearAlgebra::solve_linear_system(self, &b)
    }

    fn inv(&self) -> KwaversResult<Self> {
        LinearAlgebra::matrix_inverse(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<f64>, Self)> {
        EigenDecomposition::eigendecomposition(self)
    }
}

impl LinearAlgebraExt<Complex<f64>> for Array2<Complex<f64>> {
    fn solve_into(&self, b: Array1<Complex<f64>>) -> KwaversResult<Array1<Complex<f64>>> {
        ComplexLinearAlgebra::solve_linear_system_complex(self, &b)
    }

    fn inv(&self) -> KwaversResult<Self> {
        ComplexLinearAlgebra::matrix_inverse_complex(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<Complex<f64>>, Self)> {
        let (eigenvalues, eigenvectors) =
            EigenDecomposition::hermitian_eigendecomposition_complex(self)?;
        Ok((
            eigenvalues.mapv(|lambda| Complex::new(lambda, 0.0)),
            eigenvectors,
        ))
    }
}
