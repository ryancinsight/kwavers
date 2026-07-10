use super::covariance::{
    create_diagonal_dominant_covariance, create_identity_covariance,
    create_rank_deficient_covariance, create_test_covariance, TestCovarianceBuilder,
};
use super::steering::create_steering_vector;
use approx::assert_relative_eq;
use std::f64::consts::PI;

#[test]
fn test_covariance_is_hermitian() {
    let cov = create_test_covariance(8, 0.2, 0.1);

    for i in 0..8 {
        for j in 0..8 {
            let r_ij = cov[[i, j]];
            let r_ji_conj = cov[[j, i]].conj();
            assert_relative_eq!(r_ij.re, r_ji_conj.re, epsilon = 1e-12);
            assert_relative_eq!(r_ij.im, r_ji_conj.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_covariance_is_positive_definite() {
    let cov = create_test_covariance(8, 0.2, 0.1);

    for i in 0..8 {
        assert!(cov[[i, i]].re > 0.0);
        assert_relative_eq!(cov[[i, i]].im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn test_steering_vector_unit_elements() {
    let steering = create_steering_vector(8, 0.0);

    for &s in steering.iter() {
        assert_relative_eq!(s.norm(), 1.0, epsilon = 1e-12);
    }
}

#[test]
fn test_steering_broadside_is_ones() {
    let steering = create_steering_vector(8, 0.0);

    for &s in steering.iter() {
        assert_relative_eq!(s.re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(s.im, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn test_builder_pattern() {
    let cov = TestCovarianceBuilder::new(4)
        .with_decay(0.3)
        .with_diagonal_loading(0.05)
        .build();

    assert_eq!(cov.shape()[0], 4);
    assert_eq!(cov.shape()[1], 4);

    for i in 0..4 {
        assert!(cov[[i, i]].re >= 1.05);
    }
}

#[test]
fn test_diagonal_dominant_covariance() {
    let cov = create_diagonal_dominant_covariance(4, 0.1);

    for i in 0..4 {
        assert_relative_eq!(cov[[i, i]].re, 1.0, epsilon = 1e-12);
    }

    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                assert!(cov[[i, j]].norm() <= 0.1);
            }
        }
    }
}

#[test]
fn test_identity_covariance() {
    let cov = create_identity_covariance(4);

    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                assert_relative_eq!(cov[[i, j]].re, 1.0, epsilon = 1e-12);
                assert_relative_eq!(cov[[i, j]].im, 0.0, epsilon = 1e-12);
            } else {
                assert_relative_eq!(cov[[i, j]].re, 0.0, epsilon = 1e-12);
                assert_relative_eq!(cov[[i, j]].im, 0.0, epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn test_rank_deficient_covariance() {
    let cov = create_rank_deficient_covariance(4, 2);

    for i in 0..4 {
        for j in 0..4 {
            let r_ij = cov[[i, j]];
            let r_ji_conj = cov[[j, i]].conj();
            assert_relative_eq!(r_ij.re, r_ji_conj.re, epsilon = 1e-12);
            assert_relative_eq!(r_ij.im, r_ji_conj.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_angle_conversion() {
    use super::angle;
    assert_relative_eq!(angle::deg_to_rad(0.0), 0.0, epsilon = 1e-12);
    assert_relative_eq!(angle::deg_to_rad(90.0), PI / 2.0, epsilon = 1e-12);
    assert_relative_eq!(angle::deg_to_rad(180.0), PI, epsilon = 1e-12);

    assert_relative_eq!(angle::rad_to_deg(0.0), 0.0, epsilon = 1e-12);
    assert_relative_eq!(angle::rad_to_deg(PI / 2.0), 90.0, epsilon = 1e-12);
    assert_relative_eq!(angle::rad_to_deg(PI), 180.0, epsilon = 1e-12);
}

#[test]
fn test_angle_constants() {
    use super::angle;
    assert_eq!(angle::BROADSIDE, 0.0);
    assert_eq!(angle::ENDFIRE_POS, PI / 2.0);
    assert_eq!(angle::ENDFIRE_NEG, -PI / 2.0);
}
