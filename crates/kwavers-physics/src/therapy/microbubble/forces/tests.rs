use super::radiation::{
    calculate_drag_force, calculate_primary_bjerknes_force, Direction3D, PressureGradient3D,
    RadiationForce,
};
use super::streaming::calculate_acoustic_streaming_velocity;
use crate::therapy::microbubble::Velocity3D;
use aequitas::systems::si::quantities::{
    Dimensionless, Force, Frequency, Length, PressureGradient, Velocity,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;

fn gradient(x: f64, y: f64, z: f64) -> PressureGradient3D {
    PressureGradient3D::new(
        PressureGradient::from_base(x),
        PressureGradient::from_base(y),
        PressureGradient::from_base(z),
    )
}

#[test]
fn test_radiation_force_magnitude() {
    let force = RadiationForce::new(
        Force::from_base(3.0),
        Force::from_base(4.0),
        Force::from_base(0.0),
    );
    assert_eq!(force.magnitude().into_base(), 5.0);
}

#[test]
fn test_radiation_force_normalized() {
    let force = RadiationForce::new(
        Force::from_base(3.0),
        Force::from_base(4.0),
        Force::from_base(0.0),
    );
    let norm = force.normalized();
    assert!((norm.x * norm.x + norm.y * norm.y + norm.z * norm.z - 1.0).abs() < 1e-10);
    assert!((norm.x - 0.6).abs() < 1e-10);
    assert!((norm.y - 0.8).abs() < 1e-10);
}

#[test]
fn test_primary_bjerknes_force_basic() {
    let radius = Length::from_base(1.0e-6);
    let force = calculate_primary_bjerknes_force(radius, radius, gradient(1e5, 0.0, 0.0)).unwrap();

    assert!(force.fx.into_base() < 0.0);
    assert_eq!(force.fy.into_base(), 0.0);
    assert_eq!(force.fz.into_base(), 0.0);

    let volume = (4.0 / 3.0) * std::f64::consts::PI * 1.0e-6_f64.powi(3);
    let expected_magnitude = volume * 1e5;
    assert!((force.magnitude().into_base() - expected_magnitude).abs() < 1e-15);
}

#[test]
fn test_primary_bjerknes_expanded_bubble() {
    let radius = Length::from_base(2.0e-6);
    let force = calculate_primary_bjerknes_force(
        radius,
        Length::from_base(1.0e-6),
        gradient(1e5, 0.0, 0.0),
    )
    .unwrap();

    let volume = (4.0 / 3.0) * std::f64::consts::PI * 2.0e-6_f64.powi(3);
    let expected = -volume * 1e5;
    assert!((force.fx.into_base() - expected).abs() < 1e-10);
}

#[test]
fn test_primary_bjerknes_3d_gradient() {
    let radius = Length::from_base(1.0e-6);
    let force = calculate_primary_bjerknes_force(radius, radius, gradient(1e5, 2e5, 3e5)).unwrap();

    let volume = (4.0 / 3.0) * std::f64::consts::PI * 1.0e-6_f64.powi(3);
    assert!((force.fx.into_base() + volume * 1e5).abs() < 1e-10);
    assert!((force.fy.into_base() + volume * 2e5).abs() < 1e-10);
    assert!((force.fz.into_base() + volume * 3e5).abs() < 1e-10);
}

#[test]
fn test_streaming_velocity_zero_at_surface() {
    let r0 = Length::from_base(1.0e-6);
    let v = calculate_acoustic_streaming_velocity(
        r0,
        Velocity::from_base(10.0),
        Frequency::from_base(MHZ_TO_HZ),
        r0,
        Direction3D::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    assert_eq!(v.vx.into_base(), 0.0);
    assert_eq!(v.vy.into_base(), 0.0);
    assert_eq!(v.vz.into_base(), 0.0);
}

#[test]
fn test_streaming_velocity_far_field() {
    let r0 = Length::from_base(1.0e-6);
    let v = calculate_acoustic_streaming_velocity(
        r0,
        Velocity::from_base(10.0),
        Frequency::from_base(MHZ_TO_HZ),
        Length::from_base(10.0e-6),
        Direction3D::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    assert!(v.vx.into_base() > 0.0);
    assert_eq!(v.vy.into_base(), 0.0);
    assert_eq!(v.vz.into_base(), 0.0);

    let v_far = calculate_acoustic_streaming_velocity(
        r0,
        Velocity::from_base(10.0),
        Frequency::from_base(MHZ_TO_HZ),
        Length::from_base(20.0e-6),
        Direction3D::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    assert!(v_far.vx.into_base() < v.vx.into_base());
}

#[test]
fn test_drag_force() {
    let force = calculate_drag_force(
        Length::from_base(1.0e-6),
        Velocity3D::new(
            Velocity::from_base(1.0),
            Velocity::from_base(0.0),
            Velocity::from_base(0.0),
        ),
    )
    .unwrap();

    assert!(force.fx.into_base() < 0.0);
    assert_eq!(force.fy.into_base(), 0.0);
    assert_eq!(force.fz.into_base(), 0.0);

    let mu = kwavers_core::constants::cavitation::VISCOSITY_WATER;
    let expected = -6.0 * std::f64::consts::PI * mu * 1.0e-6;
    assert!((force.fx.into_base() - expected).abs() < 1e-15);
}

#[test]
fn test_force_addition() {
    let f1 = RadiationForce::new(
        Force::from_base(1.0),
        Force::from_base(2.0),
        Force::from_base(3.0),
    );
    let f2 = RadiationForce::new(
        Force::from_base(4.0),
        Force::from_base(5.0),
        Force::from_base(6.0),
    );
    let sum = f1.add(&f2);
    assert_eq!(sum.fx.into_base(), 5.0);
    assert_eq!(sum.fy.into_base(), 7.0);
    assert_eq!(sum.fz.into_base(), 9.0);
}

#[test]
fn test_force_scaling() {
    let force = RadiationForce::new(
        Force::from_base(1.0),
        Force::from_base(2.0),
        Force::from_base(3.0),
    );
    let scaled = force.scale(Dimensionless::from_base(2.0));
    assert_eq!(scaled.fx.into_base(), 2.0);
    assert_eq!(scaled.fy.into_base(), 4.0);
    assert_eq!(scaled.fz.into_base(), 6.0);
}
