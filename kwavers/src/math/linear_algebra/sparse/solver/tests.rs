//! Tests for [`IterativeSolver`]: identity and diagonal complex BiCGSTAB.

use super::{IterativeSolver, SolverConfig};
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

#[test]
fn test_bicgstab_complex_identity() {
    let mut a = CompressedSparseRowMatrix::<Complex64>::create(2, 2);
    a.set_diagonal(0, Complex64::new(1.0, 0.0));
    a.set_diagonal(1, Complex64::new(1.0, 0.0));

    let b = Array1::from_vec(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)]);

    let solver = IterativeSolver::create(SolverConfig::default());
    let x = solver.bicgstab_complex(&a, b.view(), None).unwrap();

    assert!((x[0] - Complex64::new(1.0, 1.0)).norm() < 1e-6);
    assert!((x[1] - Complex64::new(2.0, 2.0)).norm() < 1e-6);
}

#[test]
fn test_bicgstab_complex_diagonal() {
    // A = diag(2+i, 3+2i), x* = (1, 1), b = A·x*
    let mut a = CompressedSparseRowMatrix::<Complex64>::create(2, 2);
    a.set_diagonal(0, Complex64::new(2.0, 1.0));
    a.set_diagonal(1, Complex64::new(3.0, 2.0));

    let b = Array1::from_vec(vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, 2.0)]);

    let solver = IterativeSolver::create(SolverConfig::default());
    let x = solver.bicgstab_complex(&a, b.view(), None).unwrap();

    assert!((x[0] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
    assert!((x[1] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
}
