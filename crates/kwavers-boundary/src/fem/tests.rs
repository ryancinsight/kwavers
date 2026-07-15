use super::manager::FemBoundaryManager;
use kwavers_math::fft::Complex64;
use kwavers_math::linear_algebra::sparse::CompressedSparseRowMatrix;
use leto::Array1;

#[test]
fn test_fem_boundary_manager_creation() {
    let manager = FemBoundaryManager::new();
    assert!(manager.is_empty());
    assert_eq!(manager.len(), 0);
}

#[test]
fn test_dirichlet_boundary_condition() {
    let mut manager = FemBoundaryManager::new();

    // Add Dirichlet BC: u[0] = 1.0 + 0.5i
    manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.5))]);

    let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
    let mut mass = CompressedSparseRowMatrix::create(3, 3);
    stiffness.add_value(0, 0, Complex64::new(4.0, 0.0));
    stiffness.add_value(0, 1, Complex64::new(-1.0, 0.0));
    mass.add_value(0, 0, Complex64::new(2.0, 0.0));
    mass.add_value(0, 1, Complex64::new(0.25, 0.0));
    let mut rhs = Array1::zeros([3]);

    manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
        .unwrap();

    assert_eq!(rhs[0], Complex64::new(1.0, 0.5));
    assert_eq!(stiffness.get_diagonal(0), Complex64::new(1.0, 0.0));
    let (stiffness_values, stiffness_cols) = stiffness.get_row(0);
    assert_eq!(stiffness_cols, &[0]);
    assert_eq!(stiffness_values, &[Complex64::new(1.0, 0.0)]);
    let (mass_values, mass_cols) = mass.get_row(0);
    assert!(mass_values.is_empty());
    assert!(mass_cols.is_empty());
}

#[test]
fn test_neumann_boundary_condition() {
    let mut manager = FemBoundaryManager::new();

    // Add Neumann BC: ∂u/∂n[1] = 2.0
    manager.add_neumann(vec![(1, Complex64::new(2.0, 0.0))]);

    let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
    let mut mass = CompressedSparseRowMatrix::create(3, 3);
    let mut rhs = Array1::from_vec(
        3,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
        .unwrap();

    assert_eq!(rhs[1], Complex64::new(3.0, 0.0));
}

#[test]
fn test_robin_boundary_condition() {
    let mut manager = FemBoundaryManager::new();

    // Add Robin BC: ∂u/∂n + 0.5*u[2] = 3.0
    manager.add_robin(vec![(2, 0.5, Complex64::new(3.0, 0.0))]);

    let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
    let mut mass = CompressedSparseRowMatrix::create(3, 3);
    let mut rhs = Array1::from_vec(
        3,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    stiffness.set_diagonal(2, Complex64::new(2.0, 0.0));

    manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
        .unwrap();

    assert_eq!(stiffness.get_diagonal(2), Complex64::new(2.5, 0.0));
    assert_eq!(rhs[2], Complex64::new(3.0, 0.0));
}

#[test]
fn test_radiation_boundary_condition() {
    let mut manager = FemBoundaryManager::new();

    manager.add_radiation(vec![0]);

    let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
    let mut mass = CompressedSparseRowMatrix::create(3, 3);
    let mut rhs = Array1::zeros([3]);

    stiffness.set_diagonal(0, Complex64::new(1.0, 0.0));

    manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 2.0)
        .unwrap();

    // K_ii += -i*k = -i*2.0 → (1.0, -2.0)
    let expected = Complex64::new(1.0, -2.0);
    assert_eq!(stiffness.get_diagonal(0), expected);
}

#[test]
fn test_rejects_out_of_bounds_boundary_node_before_csr_access() {
    let mut manager = FemBoundaryManager::new();
    manager.add_neumann(vec![(3, Complex64::new(1.0, 0.0))]);

    let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
    let mut mass = CompressedSparseRowMatrix::create(3, 3);
    let mut rhs = Array1::zeros([3]);

    let error = manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
        .unwrap_err();

    assert!(error.to_string().contains("node index 3"));
    assert_eq!(rhs, Array1::zeros([3]));
}

#[test]
fn test_rejects_inconsistent_system_dimensions() {
    let mut manager = FemBoundaryManager::new();
    manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);

    let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
    let mut mass = CompressedSparseRowMatrix::create(2, 2);
    let mut rhs = Array1::zeros([3]);

    let error = manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
        .unwrap_err();

    assert!(error.to_string().contains("Mass matrix shape"));
}

#[test]
fn test_rejects_nonfinite_robin_and_radiation_parameters() {
    let mut manager = FemBoundaryManager::new();
    manager.add_robin(vec![(0, f64::NAN, Complex64::new(1.0, 0.0))]);

    let mut stiffness = CompressedSparseRowMatrix::create(2, 2);
    let mut mass = CompressedSparseRowMatrix::create(2, 2);
    let mut rhs = Array1::zeros([2]);

    let robin_error = manager
        .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
        .unwrap_err();
    assert!(robin_error.to_string().contains("alpha"));

    let mut radiation = FemBoundaryManager::new();
    radiation.add_radiation(vec![0]);
    let radiation_error = radiation
        .apply_all(&mut stiffness, &mut mass, &mut rhs, f64::INFINITY)
        .unwrap_err();
    assert!(radiation_error.to_string().contains("wavenumber"));
}
