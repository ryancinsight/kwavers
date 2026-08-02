use kwavers_core::constants::numerical::MHZ_TO_HZ;
use leto::Array3;

use crate::therapy::microbubble_dynamics::{DrugLoadingMode, DrugPayload};
use aequitas::systems::si::quantities::{Length, Pressure, PressureGradient, PressureRate, Time};
use kwavers_physics::therapy::microbubble::{
    MarmottantShellProperties, MicrobubbleState, Position3D, PressureGradient3D,
};

use super::*;
use kwavers_core::constants::numerical::TWO_PI;

fn gradient(x: f64, y: f64, z: f64) -> PressureGradient3D {
    PressureGradient3D::new(
        PressureGradient::from_base(x),
        PressureGradient::from_base(y),
        PressureGradient::from_base(z),
    )
}

#[test]
fn test_create_service() {
    let position = Position3D::zero();
    let bubble = MicrobubbleState::sono_vue(position).unwrap();
    let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();

    assert!(service.keller_miksis.params().r0 > 0.0);
}

#[test]
fn test_update_bubble_dynamics_basic() {
    let position = Position3D::zero();
    let mut bubble = MicrobubbleState::sono_vue(position).unwrap();
    let mut shell = MarmottantShellProperties::sono_vue(bubble.radius_equilibrium).unwrap();
    let volume = bubble.volume().into_base();
    let mut drug = DrugPayload::new(50.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();

    let result = service.update_bubble_dynamics(
        &mut bubble,
        &mut shell,
        &mut drug,
        Pressure::from_base(1e5),
        gradient(1e5, 0.0, 0.0),
        PressureRate::from_base(0.0),
        Time::from_base(0.0),
        Time::from_base(1e-6),
    );

    result.unwrap();
    assert!(bubble.radius.into_base() > 0.0);
    assert!(bubble.time.into_base() > 0.0);
}

#[test]
fn test_radiation_force_moves_bubble() {
    // Primary Bjerknes force: F = -(4π/3)R³ · ∇P.
    // With ∇P = (1e6, 0, 0) Pa/m and R > 0, F_x < 0 → bubble drifts toward −x.
    //
    // dt must be ≤ 1 µs to remain within the K-M adaptive integrator's
    // convergence domain; 1e-5 s is 20× the acoustic half-period and causes
    // ConvergenceFailure.  100 steps × 1e-6 s = 100 µs gives measurable drift.
    let position = Position3D::zero();
    let mut bubble = MicrobubbleState::sono_vue(position).unwrap();
    let mut shell = MarmottantShellProperties::sono_vue(bubble.radius_equilibrium).unwrap();
    let volume = bubble.volume().into_base();
    let mut drug = DrugPayload::new(0.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();

    let initial_x = bubble.position.x.into_base(); // 0.0 from Position3D::zero()
    let pressure_gradient = gradient(1e6, 0.0, 0.0);

    const N_STEPS: usize = 100;
    const DT: f64 = 1e-6;

    for i in 0..N_STEPS {
        let t = i as f64 * DT;
        service
            .update_bubble_dynamics(
                &mut bubble,
                &mut shell,
                &mut drug,
                Pressure::from_base(1e5),
                pressure_gradient,
                PressureRate::from_base(0.0),
                Time::from_base(t),
                Time::from_base(DT),
            )
            .unwrap();
    }

    // Bjerknes force is in −x: bubble must have drifted left.
    assert!(
        bubble.position.x.into_base() < initial_x,
        "radiation force should push bubble in −x direction; got position.x = {}",
        bubble.position.x.into_base(),
    );
}

#[test]
fn test_drug_release_over_time() {
    let position = Position3D::zero();
    let mut bubble = MicrobubbleState::sono_vue(position).unwrap();
    let mut shell = MarmottantShellProperties::sono_vue(bubble.radius_equilibrium).unwrap();
    let volume = bubble.volume().into_base();
    let mut drug = DrugPayload::new(100.0, volume, DrugLoadingMode::ShellEmbedded, 0.1).unwrap();

    let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();

    let initial_drug = drug.concentration;

    for i in 0..10 {
        let t = i as f64 * 1e-5;
        service
            .update_bubble_dynamics(
                &mut bubble,
                &mut shell,
                &mut drug,
                Pressure::from_base(0.0),
                gradient(0.0, 0.0, 0.0),
                PressureRate::from_base(0.0),
                Time::from_base(t),
                Time::from_base(1e-5),
            )
            .unwrap();
    }

    assert!(drug.concentration <= initial_drug);
    assert!(bubble.drug_released_total.into_base() >= 0.0);
}

#[test]
fn test_shell_rupture_detection() {
    let position = Position3D::zero();
    let mut bubble = MicrobubbleState::sono_vue(position).unwrap();
    let mut shell = MarmottantShellProperties::sono_vue(bubble.radius_equilibrium).unwrap();
    let volume = bubble.volume().into_base();
    let mut drug = DrugPayload::new(0.0, volume, DrugLoadingMode::ShellEmbedded, 0.01).unwrap();

    let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();

    bubble.radius = Length::from_base(bubble.radius_equilibrium.into_base() * 2.0);

    service
        .update_bubble_dynamics(
            &mut bubble,
            &mut shell,
            &mut drug,
            Pressure::from_base(0.0),
            gradient(0.0, 0.0, 0.0),
            PressureRate::from_base(0.0),
            Time::from_base(0.0),
            Time::from_base(1e-6),
        )
        .unwrap();

    assert!(shell.is_ruptured());
    assert!(bubble.shell_is_ruptured);
}

#[test]
fn test_pressure_time_derivative_changes_dynamics() {
    let position = Position3D::zero();
    let acoustic_pressure = 1e5_f64;
    let frequency = MHZ_TO_HZ;
    let dp_dt_nonzero = acoustic_pressure * TWO_PI * frequency;

    let mut bubble_zero = MicrobubbleState::sono_vue(position).unwrap();
    let mut shell_zero =
        MarmottantShellProperties::sono_vue(bubble_zero.radius_equilibrium).unwrap();
    let volume = bubble_zero.volume().into_base();
    let mut drug_zero = DrugPayload::new(0.0, volume, DrugLoadingMode::ShellEmbedded, 0.0).unwrap();
    let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble_zero).unwrap();
    service
        .update_bubble_dynamics(
            &mut bubble_zero,
            &mut shell_zero,
            &mut drug_zero,
            Pressure::from_base(acoustic_pressure),
            gradient(0.0, 0.0, 0.0),
            PressureRate::from_base(0.0),
            Time::from_base(0.0),
            Time::from_base(1e-7),
        )
        .unwrap();

    let mut bubble_nonzero = MicrobubbleState::sono_vue(position).unwrap();
    let mut shell_nonzero =
        MarmottantShellProperties::sono_vue(bubble_nonzero.radius_equilibrium).unwrap();
    let mut drug_nonzero =
        DrugPayload::new(0.0, volume, DrugLoadingMode::ShellEmbedded, 0.0).unwrap();
    service
        .update_bubble_dynamics(
            &mut bubble_nonzero,
            &mut shell_nonzero,
            &mut drug_nonzero,
            Pressure::from_base(acoustic_pressure),
            gradient(0.0, 0.0, 0.0),
            PressureRate::from_base(dp_dt_nonzero),
            Time::from_base(0.0),
            Time::from_base(1e-7),
        )
        .unwrap();

    assert_ne!(
        bubble_zero.radius, bubble_nonzero.radius,
        "Non-zero dP/dt must change bubble dynamics (radiation-damping term)"
    );
}

#[test]
fn test_sample_acoustic_field() {
    let mut pressure = Array3::zeros((10, 10, 10));
    pressure[[5, 5, 5]] = 1e5;

    let position = Position3D::new(
        Length::from_base(0.005),
        Length::from_base(0.005),
        Length::from_base(0.005),
    );
    let grid_spacing = (
        Length::from_base(0.001),
        Length::from_base(0.001),
        Length::from_base(0.001),
    );

    let (p, _grad) = sample_acoustic_field_at_position(&position, &pressure, grid_spacing).unwrap();

    assert_eq!(p.into_base(), 1e5);
}

#[test]
fn test_effective_mass() {
    let radius = 1e-6;
    let mass = MicrobubbleDynamicsService::effective_bubble_mass(radius);

    assert!(mass > 0.0);

    let mass_2r = MicrobubbleDynamicsService::effective_bubble_mass(2.0 * radius);
    assert!((mass_2r / mass - 8.0).abs() < 0.01);
}
