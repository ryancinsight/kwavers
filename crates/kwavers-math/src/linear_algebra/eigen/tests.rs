use super::decomposition::EigenDecomposition;
use eunomia::Complex64 as Complex;
use ndarray::Array2;

#[test]
fn test_real_symmetric_eigendecomposition() {
    let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let (eigenvals, eigenvecs) = EigenDecomposition::eigendecomposition(&matrix).unwrap();

    assert!((eigenvals[0] - 3.0).abs() < 1e-6);
    assert!((eigenvals[1] - 1.0).abs() < 1e-6);

    for i in 0..2 {
        let lambda = eigenvals[i];
        let v = eigenvecs.column(i);
        let av = matrix.dot(&v.to_owned());
        for j in 0..2 {
            assert!((av[j] - lambda * v[j]).abs() < 1e-6);
        }
    }
}

/// Previously broken: a real symmetric matrix with **unequal diagonal**
/// elements drove the malformed `else`-branch Jacobi angle (the equal-diagonal
/// case took the correct π/4 branch and hid the bug). Verify exact eigenvalues,
/// the eigen-relation `Av = λv`, and the reconstruction `A = VΛVᵀ`.
#[test]
fn test_real_symmetric_unequal_diagonal() {
    // [[4,1],[1,2]] ⇒ λ = 3 ± √2.
    let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 2.0]).unwrap();
    let (vals, vecs) = EigenDecomposition::eigendecomposition(&matrix).unwrap();

    let hi = 3.0 + 2.0_f64.sqrt();
    let lo = 3.0 - 2.0_f64.sqrt();
    assert!((vals[0] - hi).abs() < 1e-8, "λ_max {} vs {hi}", vals[0]);
    assert!((vals[1] - lo).abs() < 1e-8, "λ_min {} vs {lo}", vals[1]);

    // Av = λv for each mode.
    for i in 0..2 {
        let v = vecs.column(i).to_owned();
        let av = matrix.dot(&v);
        for j in 0..2 {
            assert!((av[j] - vals[i] * v[j]).abs() < 1e-8);
        }
    }
    // Reconstruction A = V Λ Vᵀ.
    let lambda = Array2::from_diag(&vals);
    let recon = vecs.dot(&lambda).dot(&vecs.t());
    for i in 0..2 {
        for j in 0..2 {
            assert!((recon[[i, j]] - matrix[[i, j]]).abs() < 1e-8);
        }
    }
}

/// A 3×3 real symmetric matrix with all-distinct, unequal-diagonal structure
/// reconstructs exactly — exercising the delegated solver beyond 2×2.
#[test]
fn test_real_symmetric_3x3_reconstruction() {
    let matrix =
        Array2::from_shape_vec((3, 3), vec![6.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 4.0]).unwrap();
    let (vals, vecs) = EigenDecomposition::eigendecomposition(&matrix).unwrap();
    // Trace and determinant invariants.
    let trace = matrix[[0, 0]] + matrix[[1, 1]] + matrix[[2, 2]];
    assert!(
        (vals.sum() - trace).abs() < 1e-8,
        "Σλ {} vs tr {trace}",
        vals.sum()
    );
    // Reconstruction.
    let lambda = Array2::from_diag(&vals);
    let recon = vecs.dot(&lambda).dot(&vecs.t());
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (recon[[i, j]] - matrix[[i, j]]).abs() < 1e-7,
                "recon[{i}][{j}] {} vs {}",
                recon[[i, j]],
                matrix[[i, j]]
            );
        }
    }
    // Descending order.
    assert!(vals[0] >= vals[1] && vals[1] >= vals[2]);
}

#[test]
fn test_hermitian_eigendecomposition_identity() {
    let n = 3;
    let identity = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        }
    });

    let (eigenvals, _eigenvecs) =
        EigenDecomposition::hermitian_eigendecomposition_complex(&identity).unwrap();

    for i in 0..n {
        assert!((eigenvals[i] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_hermitian_eigendecomposition_diagonal() {
    let diag_vals = vec![5.0, 3.0, 1.0];
    let matrix = Array2::from_shape_fn((3, 3), |(i, j)| {
        if i == j {
            Complex::new(diag_vals[i], 0.0)
        } else {
            Complex::new(0.0, 0.0)
        }
    });

    let (eigenvals, eigenvecs) =
        EigenDecomposition::hermitian_eigendecomposition_complex(&matrix).unwrap();

    let mut expected = diag_vals.clone();
    expected.sort_by(|a, b| b.total_cmp(a));

    for i in 0..3 {
        assert!(
            (eigenvals[i] - expected[i]).abs() < 1e-10,
            "Eigenvalue mismatch at {}: expected {}, got {}",
            i,
            expected[i],
            eigenvals[i]
        );
    }

    let lambda_diag = Array2::from_diag(&eigenvals.mapv(|x| Complex::new(x, 0.0)));
    let vdag: Array2<Complex> = eigenvecs
        .t()
        .mapv(|z| z.conj())
        .into_dimensionality()
        .unwrap();
    let v_lambda = eigenvecs.dot(&lambda_diag);
    let reconstructed: Array2<Complex> = v_lambda.dot(&vdag);

    for i in 0..3 {
        for j in 0..3 {
            assert!((matrix[[i, j]] - reconstructed[[i, j]]).norm() < 1e-10);
        }
    }
}

#[test]
fn test_hermitian_eigendecomposition_2x2() {
    // Analytical eigenvalues: det(H − λI) = λ² − 5λ + 4 = 0 → λ ∈ {4, 1}
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let (eigenvals, eigenvecs) =
        EigenDecomposition::hermitian_eigendecomposition_complex(&matrix).unwrap();

    assert!(
        (eigenvals[0] - 4.0).abs() < 1e-10,
        "Large eigenvalue mismatch: expected 4.0, got {}",
        eigenvals[0]
    );
    assert!(
        (eigenvals[1] - 1.0).abs() < 1e-10,
        "Small eigenvalue mismatch: expected 1.0, got {}",
        eigenvals[1]
    );

    for i in 0..2 {
        let lambda = eigenvals[i];
        let v = eigenvecs.column(i).to_owned();
        let hv = matrix.dot(&v);
        for j in 0..2 {
            assert!((hv[j] - lambda * v[j]).norm() < 1e-10);
        }
    }

    let vdag: Array2<Complex> = eigenvecs
        .t()
        .mapv(|z| z.conj())
        .into_dimensionality()
        .unwrap();
    let vdag_v: Array2<Complex> = vdag.dot(&eigenvecs);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((vdag_v[[i, j]].re - expected).abs() < 1e-10);
            assert!(vdag_v[[i, j]].im.abs() < 1e-10);
        }
    }
}

#[test]
fn test_hermitian_eigendecomposition_non_hermitian_rejected() {
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 1.0), // Not conjugate of (1,1)
            Complex::new(2.0, 0.0),
        ],
    )
    .unwrap();

    let result = EigenDecomposition::hermitian_eigendecomposition_complex(&matrix);
    assert!(result.is_err());
}

#[test]
fn test_hermitian_eigendecomposition_real_eigenvalues() {
    let matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            Complex::new(4.0, 0.0),
            Complex::new(1.0, -2.0),
            Complex::new(2.0, 1.0),
            Complex::new(1.0, 2.0),
            Complex::new(5.0, 0.0),
            Complex::new(-1.0, 3.0),
            Complex::new(2.0, -1.0),
            Complex::new(-1.0, -3.0),
            Complex::new(6.0, 0.0),
        ],
    )
    .unwrap();

    let (eigenvals, _) = EigenDecomposition::hermitian_eigendecomposition_complex(&matrix).unwrap();

    for &lambda in eigenvals.iter() {
        assert!(lambda.is_finite());
    }
}
