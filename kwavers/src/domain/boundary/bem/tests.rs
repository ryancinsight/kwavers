use super::manager::BemBoundaryManager;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

#[test]
fn test_bem_boundary_manager_creation() {
    let manager = BemBoundaryManager::new();
    assert!(manager.is_empty());
    assert_eq!(manager.len(), 0);
}

#[test]
fn test_dirichlet_boundary_condition() {
    let mut manager = BemBoundaryManager::new();
    manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.5))]);

    let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut boundary_values = Array1::zeros(3);

    manager
        .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 1.0)
        .unwrap();

    assert_eq!(boundary_values[0], Complex64::new(1.0, 0.5));
    assert_eq!(manager.len(), 1);
}

#[test]
fn test_neumann_boundary_condition() {
    let mut manager = BemBoundaryManager::new();
    manager.add_neumann(vec![(1, Complex64::new(2.0, 0.0))]);

    let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut boundary_values = Array1::zeros(3);

    manager
        .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 1.0)
        .unwrap();

    assert_eq!(boundary_values[1], Complex64::new(2.0, 0.0));
    assert_eq!(manager.len(), 1);
}

#[test]
fn test_robin_boundary_condition() {
    let mut manager = BemBoundaryManager::new();
    manager.add_robin(vec![(2, 0.5, Complex64::new(3.0, 0.0))]);

    let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut boundary_values = Array1::from_vec(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);

    h_matrix.set_diagonal(2, Complex64::new(2.0, 0.0));

    manager
        .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 1.0)
        .unwrap();

    assert_eq!(h_matrix.get_diagonal(2), Complex64::new(2.5, 0.0));
    assert_eq!(boundary_values[2], Complex64::new(3.0, 0.0));
}

#[test]
fn test_radiation_boundary_condition() {
    let mut manager = BemBoundaryManager::new();
    manager.add_radiation(vec![0]);

    let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
    let mut boundary_values = Array1::zeros(3);

    h_matrix.set_diagonal(0, Complex64::new(1.0, 0.0));

    manager
        .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 2.0)
        .unwrap();

    // H_ii += −i·k = (0, −2) → total (1, −2)
    let expected = Complex64::new(1.0, -2.0);
    assert_eq!(h_matrix.get_diagonal(0), expected);
}
