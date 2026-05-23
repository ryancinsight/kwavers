use super::radiation::{calculate_drag_force, calculate_primary_bjerknes_force, RadiationForce};
use super::streaming::calculate_acoustic_streaming_velocity;
use crate::core::constants::numerical::MHZ_TO_HZ;

#[test]
fn test_radiation_force_magnitude() {
    let force = RadiationForce::new(3.0, 4.0, 0.0);
    assert_eq!(force.magnitude(), 5.0);
}

#[test]
fn test_radiation_force_normalized() {
    let force = RadiationForce::new(3.0, 4.0, 0.0);
    let norm = force.normalized();
    assert!((norm.magnitude() - 1.0).abs() < 1e-10);
    assert!((norm.fx - 0.6).abs() < 1e-10);
    assert!((norm.fy - 0.8).abs() < 1e-10);
}

#[test]
fn test_primary_bjerknes_force_basic() {
    let radius = 1.0e-6;
    let r0 = 1.0e-6;
    let grad_p = (1e5, 0.0, 0.0);

    let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();

    assert!(force.fx < 0.0);
    assert_eq!(force.fy, 0.0);
    assert_eq!(force.fz, 0.0);

    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    let expected_magnitude = volume * 1e5;
    assert!((force.magnitude() - expected_magnitude).abs() < 1e-15);
}

#[test]
fn test_primary_bjerknes_expanded_bubble() {
    let radius = 2.0e-6;
    let r0 = 1.0e-6;
    let grad_p = (1e5, 0.0, 0.0);

    let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();

    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    let expected = -volume * 1e5;
    assert!((force.fx - expected).abs() < 1e-10);
}

#[test]
fn test_primary_bjerknes_3d_gradient() {
    let radius = 1.0e-6;
    let r0 = 1.0e-6;
    let grad_p = (1e5, 2e5, 3e5);

    let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();

    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    assert!((force.fx + volume * 1e5).abs() < 1e-10);
    assert!((force.fy + volume * 2e5).abs() < 1e-10);
    assert!((force.fz + volume * 3e5).abs() < 1e-10);
}

#[test]
fn test_streaming_velocity_zero_at_surface() {
    let r0 = 1.0e-6;
    let v = calculate_acoustic_streaming_velocity(r0, 10.0, MHZ_TO_HZ, r0, (1.0, 0.0, 0.0)).unwrap();
    assert_eq!(v.vx, 0.0);
    assert_eq!(v.vy, 0.0);
    assert_eq!(v.vz, 0.0);
}

#[test]
fn test_streaming_velocity_far_field() {
    let r0 = 1.0e-6;
    let v =
        calculate_acoustic_streaming_velocity(r0, 10.0, MHZ_TO_HZ, 10.0 * r0, (1.0, 0.0, 0.0)).unwrap();
    assert!(v.vx > 0.0);
    assert_eq!(v.vy, 0.0);
    assert_eq!(v.vz, 0.0);

    let v_far =
        calculate_acoustic_streaming_velocity(r0, 10.0, 1e6, 20.0 * r0, (1.0, 0.0, 0.0)).unwrap();
    assert!(v_far.vx < v.vx);
}

#[test]
fn test_drag_force() {
    let radius = 1.0e-6;
    let force = calculate_drag_force(radius, (1.0, 0.0, 0.0)).unwrap();

    assert!(force.fx < 0.0);
    assert_eq!(force.fy, 0.0);
    assert_eq!(force.fz, 0.0);

    let mu = crate::core::constants::cavitation::VISCOSITY_WATER;
    let expected = -6.0 * std::f64::consts::PI * mu * radius;
    assert!((force.fx - expected).abs() < 1e-15);
}

#[test]
fn test_force_addition() {
    let f1 = RadiationForce::new(1.0, 2.0, 3.0);
    let f2 = RadiationForce::new(4.0, 5.0, 6.0);
    let sum = f1.add(&f2);
    assert_eq!(sum.fx, 5.0);
    assert_eq!(sum.fy, 7.0);
    assert_eq!(sum.fz, 9.0);
}

#[test]
fn test_force_scaling() {
    let force = RadiationForce::new(1.0, 2.0, 3.0);
    let scaled = force.scale(2.0);
    assert_eq!(scaled.fx, 2.0);
    assert_eq!(scaled.fy, 4.0);
    assert_eq!(scaled.fz, 6.0);
}
