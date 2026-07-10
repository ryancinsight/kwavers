use super::*;
use crate::forward::pstd::dg::basis::build_vandermonde;
use leto::Array1;

#[test]
fn legendre_derivative_endpoint_limits_are_finite() {
    let endpoint_cases = [
        (2, 1.0, 1.0, 3.0),
        (2, -1.0, 1.0, -3.0),
        (3, 1.0, 1.0, 6.0),
        (3, -1.0, -1.0, 6.0),
    ];

    for (degree, node, expected_value, expected_derivative) in endpoint_cases {
        let (value, derivative) = legendre_poly_and_deriv(degree, node);
        assert_eq!(value, expected_value);
        assert_eq!(derivative, expected_derivative);
        assert!(value.is_finite());
        assert!(derivative.is_finite());
    }
}

#[test]
fn differentiation_matrix_is_finite_on_gll_endpoints() {
    let nodes = Array1::from_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Legendre).unwrap();

    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Legendre).unwrap();

    assert!(diff.iter().all(|entry| entry.is_finite()));
}

#[test]
fn differentiation_matrix_exactly_differentiates_linear_polynomial() {
    let nodes = Array1::from_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Legendre).unwrap();
    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Legendre).unwrap();

    let constant_values = Array1::ones(nodes.len() );
    let mut constant_derivative = Array1::<f64>::zeros(nodes.len());
    leto_ops::matvec(
        &diff.view(),
        &constant_values.view(),
        &mut constant_derivative.view_mut(),
    )
    .unwrap();
    assert!(constant_derivative
        .iter()
        .all(|value: &f64| value.abs() <= 1e-12));

    let mut linear_derivative = Array1::<f64>::zeros(nodes.len());
    leto_ops::matvec(&diff.view(), &nodes.view(), &mut linear_derivative.view_mut()).unwrap();
    for value in linear_derivative.iter() {
        assert!((value - 1.0).abs() <= 1e-12);
    }
}

#[test]
fn chebyshev_differentiation_matrix_is_finite_on_endpoints() {
    let nodes = Array1::from_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Chebyshev).unwrap();

    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Chebyshev).unwrap();

    assert!(diff.iter().all(|entry| entry.is_finite()));
}

#[test]
fn chebyshev_differentiation_matrix_exactly_differentiates_quadratic() {
    let nodes = Array1::from_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Chebyshev).unwrap();
    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Chebyshev).unwrap();

    let quadratic_values = nodes.mapv(|x| x * x);
    let mut derivative = Array1::<f64>::zeros(nodes.len());
    leto_ops::matvec(
        &diff.view(),
        &quadratic_values.view(),
        &mut derivative.view_mut(),
    )
    .unwrap();

    for (actual, expected) in derivative.iter().zip(nodes.iter().map(|x| 2.0 * x)) {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn fourier_differentiation_matrix_exactly_differentiates_first_sine_mode() {
    let nodes = Array1::from_vec(3, vec![-0.5, 0.0, 0.5]).unwrap();
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Fourier).unwrap();
    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Fourier).unwrap();

    let sine_values = nodes.mapv(|x| (std::f64::consts::PI * (x + 1.0)).sin());
    let mut derivative = Array1::<f64>::zeros(nodes.len());
    leto_ops::matvec(&diff.view(), &sine_values.view(), &mut derivative.view_mut()).unwrap();

    for (actual, node) in derivative.iter().zip(nodes.iter()) {
        let expected = std::f64::consts::PI * (std::f64::consts::PI * (node + 1.0)).cos();
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn fourier_differentiation_rejects_gll_duplicate_periodic_endpoints() {
    let nodes = Array1::from_vec(3, vec![-1.0, 0.0, 1.0]).unwrap();
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Legendre).unwrap();

    let error = compute_diff_matrix(&vandermonde, &nodes, BasisType::Fourier).unwrap_err();

    assert!(format!("{error}").contains("cannot include both periodic endpoints"));
}
