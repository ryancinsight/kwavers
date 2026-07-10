use super::*;
use leto::Array2;
use eunomia::Complex64;

fn create_hermitian_2x2() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap()
}

fn create_hermitian_3x3() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (3, 3),
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap()
}

#[test]
fn test_jacobi_2x2_hermitian() {
    let matrix = create_hermitian_2x2();
    let config = EigenSolverConfig::default();

    let result = EigenSolver::jacobi_hermitian(&matrix, config).unwrap();

    assert_eq!(result.eigenvalues.len(), 2);
    assert!(result.eigenvalues[0] > result.eigenvalues[1]);

    for k in 0..2 {
        let lambda = result.eigenvalues[k];
        let v = result.eigenvectors.index_axis::<1>(1, k).unwrap().to_contiguous();

        for i in 0..2 {
            let av_i = (0..2).map(|j| matrix[[i, j]] * v[j]).sum::<Complex64>();
            let error = (av_i - lambda * v[i]).norm();
            assert!(
                error < 1.5,
                "Eigenvalue equation failed for λ[{}]: error = {}",
                k,
                error
            );
        }
    }
}

#[test]
fn test_qr_algorithm_3x3_hermitian() {
    let matrix = create_hermitian_3x3();
    let config = EigenSolverConfig::default();

    let result = EigenSolver::qr_algorithm(&matrix, config).unwrap();

    assert_eq!(result.eigenvalues.len(), 3);
    assert!(result.eigenvalues[0] > result.eigenvalues[1]);
    assert!(result.eigenvalues[1] > result.eigenvalues[2]);

    for k in 0..3 {
        let lambda = result.eigenvalues[k];
        let v = result.eigenvectors.index_axis::<1>(1, k).unwrap().to_contiguous();

        for i in 0..3 {
            let av_i = (0..3).map(|j| matrix[[i, j]] * v[j]).sum::<Complex64>();
            let error = (av_i - lambda * v[i]).norm();
            assert!(
                error < 2.0,
                "QR eigenvalue equation failed for λ[{}]: error = {}",
                k,
                error
            );
        }
    }
}

#[test]
fn test_condition_number_estimation() {
    let matrix = create_hermitian_2x2();
    let config = EigenSolverConfig {
        estimate_condition: true,
        ..Default::default()
    };

    let result = EigenSolver::jacobi_hermitian(&matrix, config).unwrap();

    let kappa = result.condition_number.unwrap();
    assert!(kappa >= 1.0, "Condition number should be >= 1");
}

#[test]
fn test_eigenvalue_sorting() {
    let matrix = create_hermitian_3x3();
    let config = EigenSolverConfig {
        sort_descending: true,
        ..Default::default()
    };

    let result = EigenSolver::qr_algorithm(&matrix, config).unwrap();

    for i in 0..result.eigenvalues.len() - 1 {
        assert!(
            result.eigenvalues[i] >= result.eigenvalues[i + 1],
            "Eigenvalues not in descending order"
        );
    }
}

#[test]
fn test_non_hermitian_matrix_rejected() {
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();

    let config = EigenSolverConfig::default();
    let result = EigenSolver::jacobi_hermitian(&matrix, config);

    assert!(result.is_err(), "Non-Hermitian matrix should be rejected");
}

#[test]
fn test_dimension_mismatch_rejected() {
    let matrix = Array2::from_shape_vec(
        (2, 3),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    let config = EigenSolverConfig::default();
    let result = EigenSolver::jacobi_hermitian(&matrix, config);

    assert!(result.is_err(), "Non-square matrix should be rejected");
}

#[test]
fn test_convergence_diagnostics() {
    let matrix = create_hermitian_2x2();
    let config = EigenSolverConfig::default();

    let result = EigenSolver::jacobi_hermitian(&matrix, config).unwrap();

    assert!(result.off_diagonal_norm < config.tolerance || result.off_diagonal_norm < 1e-8);
    assert!(result.iterations > 0);
    assert!(result.iterations < 1000);
}
