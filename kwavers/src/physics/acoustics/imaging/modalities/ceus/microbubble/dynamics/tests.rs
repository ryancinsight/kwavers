use super::*;
use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::domain::imaging::ultrasound::ceus::Microbubble;

fn test_bubble() -> Microbubble {
    Microbubble::new(2.0, 1.0, 0.5)
}

#[test]
fn test_velocity_verlet_second_order_convergence() {
    let bubble = test_bubble();
    let dt_fine = 5e-11;
    let dt_coarse = 2e-10;
    let dt_ref = 5e-12;
    let duration = 50e-9;

    let sim_base = BubbleDynamics {
        dt: dt_fine,
        ambient_pressure: ATMOSPHERIC_PRESSURE,
        liquid_density: DENSITY_WATER_NOMINAL,
        damping_coefficient: 0.1,
    };

    let p_ac = 1e3;
    let freq = MHZ_TO_HZ;

    let r_ref = BubbleDynamics {
        dt: dt_ref,
        ..sim_base
    }
    .simulate_oscillation(&bubble, p_ac, freq, duration)
    .unwrap();
    let r_fine = BubbleDynamics {
        dt: dt_fine,
        ..sim_base
    }
    .simulate_oscillation(&bubble, p_ac, freq, duration)
    .unwrap();
    let r_coarse = BubbleDynamics {
        dt: dt_coarse,
        ..sim_base
    }
    .simulate_oscillation(&bubble, p_ac, freq, duration)
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
        dt: 1e-10,
        ambient_pressure: ATMOSPHERIC_PRESSURE,
        liquid_density: DENSITY_WATER_NOMINAL,
        damping_coefficient: 0.1,
    };

    let result = sim.simulate_oscillation(&bubble, 1e3, MHZ_TO_HZ, 500e-9).unwrap();

    let r0 = bubble.radius_eq;
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
