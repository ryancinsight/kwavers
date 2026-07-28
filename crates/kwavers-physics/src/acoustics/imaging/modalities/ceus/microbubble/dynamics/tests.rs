use super::*;
use aequitas::systems::si::quantities::{Frequency, Length, MassDensity, Pressure, Time};
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_imaging::ultrasound::ceus::Microbubble;

fn test_bubble() -> Microbubble {
    Microbubble::new(
        Length::from_base(2.0e-6),
        Pressure::from_base(1_000.0),
        aequitas::systems::si::quantities::DynamicViscosity::from_base(0.5),
    )
}

#[test]
fn test_velocity_verlet_second_order_convergence() {
    let bubble = test_bubble();
    let dt_fine = 5e-11;
    let dt_coarse = 2e-10;
    let dt_ref = 5e-12;
    let duration = 50e-9;

    let sim_base = BubbleDynamics {
        dt: Time::from_base(dt_fine),
        ambient_pressure: Pressure::from_base(ATMOSPHERIC_PRESSURE),
        liquid_density: MassDensity::from_base(DENSITY_WATER_NOMINAL),
        damping_coefficient: 0.1,
    };

    let p_ac = 1e3;
    let freq = Frequency::from_base(MHZ_TO_HZ);

    let r_ref = BubbleDynamics {
        dt: Time::from_base(dt_ref),
        ..sim_base
    }
    .simulate_oscillation(
        &bubble,
        Pressure::from_base(p_ac),
        freq,
        Time::from_base(duration),
    )
    .unwrap();
    let r_fine = BubbleDynamics {
        dt: Time::from_base(dt_fine),
        ..sim_base
    }
    .simulate_oscillation(
        &bubble,
        Pressure::from_base(p_ac),
        freq,
        Time::from_base(duration),
    )
    .unwrap();
    let r_coarse = BubbleDynamics {
        dt: Time::from_base(dt_coarse),
        ..sim_base
    }
    .simulate_oscillation(
        &bubble,
        Pressure::from_base(p_ac),
        freq,
        Time::from_base(duration),
    )
    .unwrap();

    let r_ref_end = *r_ref.radius.last().unwrap();
    let r_fine_end = *r_fine.radius.last().unwrap();
    let r_coarse_end = *r_coarse.radius.last().unwrap();
    let err_fine = (r_fine_end - r_ref_end).abs();
    let err_coarse = (r_coarse_end - r_ref_end).abs();

    if err_fine > 1e-20 {
        let ratio = err_coarse / err_fine;
        assert!(
            ratio > 4.0,
            "Velocity Verlet must outperform O(dt1): err_coarse={:.3e}, err_fine={:.3e}, ratio={:.2}",
            err_coarse,
            err_fine,
            ratio
        );
    }
}

#[test]
fn test_linear_oscillation_bounded() {
    let bubble = test_bubble();
    let sim = BubbleDynamics {
        dt: Time::from_base(1e-10),
        ambient_pressure: Pressure::from_base(ATMOSPHERIC_PRESSURE),
        liquid_density: MassDensity::from_base(DENSITY_WATER_NOMINAL),
        damping_coefficient: 0.1,
    };

    let result = sim
        .simulate_oscillation(
            &bubble,
            Pressure::from_base(1e3),
            Frequency::from_base(MHZ_TO_HZ),
            Time::from_base(500e-9),
        )
        .unwrap();

    let r0 = bubble.radius_eq.into_base();
    let max_r = result
        .radius
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_r = result.radius.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!(
        (max_r - r0).abs() / r0 < 0.05,
        "Max radius deviation {:.1}% exceeds 5% at 1 kPa drive",
        100.0 * (max_r - r0).abs() / r0
    );
    assert!(
        (min_r - r0).abs() / r0 < 0.05,
        "Min radius deviation {:.1}% exceeds 5% at 1 kPa drive",
        100.0 * (min_r - r0).abs() / r0
    );
}
