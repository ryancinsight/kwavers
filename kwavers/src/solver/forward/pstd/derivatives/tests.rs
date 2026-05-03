use super::operator::SpectralDerivativeOperator;
use ndarray::Array3;
use std::f64::consts::PI;

fn create_test_operator() -> SpectralDerivativeOperator {
    SpectralDerivativeOperator::new(32, 32, 32, 0.001, 0.001, 0.001)
}

#[test]
fn test_operator_creation() {
    let op = create_test_operator();
    assert_eq!(op.nx, 32);
    assert_eq!(op.ny, 32);
    assert_eq!(op.nz, 32);
}

#[test]
fn test_derivative_sinusoidal_x() {
    let op = create_test_operator();

    let mut field = Array3::zeros([32, 32, 32]);
    let k = 2.0 * PI / (32.0 * 0.001);

    for i in 0..32 {
        let x = i as f64 * 0.001;
        for j in 0..32 {
            for l in 0..32 {
                field[[i, j, l]] = (k * x).sin();
            }
        }
    }

    let field_view = field.view();
    let deriv = op.derivative_x(&field_view).unwrap();

    let expected_center = (k * 0.016).cos() * k;
    let computed = deriv[[16, 16, 16]];
    assert!(
        (computed - expected_center).abs() < 0.01,
        "Center point error: {} vs {}",
        computed,
        expected_center
    );
}

#[test]
fn test_derivative_output() {
    let op = create_test_operator();

    let mut field = Array3::zeros([32, 32, 32]);
    for i in 0..32 {
        let x = i as f64 * 0.001;
        for j in 0..32 {
            let y = j as f64 * 0.001;
            for l in 0..32 {
                field[[i, j, l]] =
                    (-(x - 0.016).powi(2) / 0.0001).exp() * (-(y - 0.016).powi(2) / 0.0001).exp();
            }
        }
    }

    let field_view = field.view();
    let deriv_x = op.derivative_x(&field_view).unwrap();
    let deriv_y = op.derivative_y(&field_view).unwrap();
    let deriv_z = op.derivative_z(&field_view).unwrap();

    assert_eq!(deriv_x.shape(), &[32, 32, 32]);
    assert_eq!(deriv_y.shape(), &[32, 32, 32]);
    assert_eq!(deriv_z.shape(), &[32, 32, 32]);

    assert!(deriv_x.iter().all(|&x| x.is_finite()));
    assert!(deriv_y.iter().all(|&y| y.is_finite()));
    assert!(deriv_z.iter().all(|&z| z.is_finite()));

    let center_val = deriv_x[[16, 16, 16]];
    assert!(center_val.abs() < 1.0, "Derivative values seem reasonable");
}

#[test]
fn test_derivatives_all_axes() {
    let op = create_test_operator();

    let field = Array3::from_elem([32, 32, 32], 5.0);
    let field_view = field.view();

    let dx = op.derivative_x(&field_view).unwrap();
    let dy = op.derivative_y(&field_view).unwrap();
    let dz = op.derivative_z(&field_view).unwrap();

    assert!(dx.iter().all(|&x| x.abs() < 1e-10));
    assert!(dy.iter().all(|&y| y.abs() < 1e-10));
    assert!(dz.iter().all(|&z| z.abs() < 1e-10));
}

#[test]
fn test_invalid_field_size() {
    let op = create_test_operator();
    let field = Array3::zeros([16, 32, 32]);
    let field_view = field.view();

    let result = op.derivative_x(&field_view);
    assert!(result.is_err());
}
