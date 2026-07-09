use super::*;
use crate::forward::pstd::dg::basis::build_vandermonde;
use leto::{
    /* arr1 -- no leto equivalent */,
    Array1,
};

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
    let nodes = arr1(&[-1.0, 0.0, 1.0]);
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Legendre).unwrap();

    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Legendre).unwrap();

    assert!(diff.iter().all(|entry| entry.is_finite()));
}

#[test]
fn differentiation_matrix_exactly_differentiates_linear_polynomial() {
    let nodes = arr1(&[-1.0, 0.0, 1.0]);
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Legendre).unwrap();
    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Legendre).unwrap();

    let constant_values = Array1::ones(nodes.len());
    let constant_derivative = diff.dot(&constant_values);
    assert!(constant_derivative
        .iter()
        .all(|value: &f64| value.abs() <= 1e-12));

    let linear_derivative = diff.dot(&nodes);
    for value in linear_derivative {
        assert!((value - 1.0).abs() <= 1e-12);
    }
}

#[test]
fn chebyshev_differentiation_matrix_is_finite_on_endpoints() {
    let nodes = arr1(&[-1.0, 0.0, 1.0]);
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Chebyshev).unwrap();

    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Chebyshev).unwrap();

    assert!(diff.iter().all(|entry| entry.is_finite()));
}

#[test]
fn chebyshev_differentiation_matrix_exactly_differentiates_quadratic() {
    let nodes = arr1(&[-1.0, 0.0, 1.0]);
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Chebyshev).unwrap();
    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Chebyshev).unwrap();

    let quadratic_values = nodes.mapv(|x| x * x);
    let derivative = diff.dot(&quadratic_values);

    for (actual, expected) in derivative.iter().zip(nodes.iter().map(|x| 2.0 * x)) {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn fourier_differentiation_matrix_exactly_differentiates_first_sine_mode() {
    let nodes = arr1(&[-0.5, 0.0, 0.5]);
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Fourier).unwrap();
    let diff = compute_diff_matrix(&vandermonde, &nodes, BasisType::Fourier).unwrap();

    let sine_values = nodes.mapv(|x| (std::f64::consts::PI * (x + 1.0)).sin());
    let derivative = diff.dot(&sine_values);

    for (actual, node) in derivative.iter().zip(nodes.iter()) {
        let expected = std::f64::consts::PI * (std::f64::consts::PI * (node + 1.0)).cos();
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn fourier_differentiation_rejects_gll_duplicate_periodic_endpoints() {
    let nodes = arr1(&[-1.0, 0.0, 1.0]);
    let vandermonde = build_vandermonde(&nodes, 2, BasisType::Legendre).unwrap();

    let error = compute_diff_matrix(&vandermonde, &nodes, BasisType::Fourier).unwrap_err();

    assert!(format!("{error}").contains("cannot include both periodic endpoints"));
}
