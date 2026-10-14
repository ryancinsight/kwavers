//! Composed first-law regressions with outward-positive conductive power.

use super::{update_temperature_energy_balance, EnergyBalanceCalculator};
use crate::acoustics::bubble_dynamics::{BubbleParameters, BubbleState};
use aequitas::systems::si::{
    quantities::{Power, Pressure},
    units::{Kelvin, Pascal, Watt},
};
use kwavers_core::constants::GAS_CONSTANT;
use std::f64::consts::PI;

fn diagnostics() -> tracing::subscriber::DefaultGuard {
    // Thread-scoped capture avoids changing the process-global subscriber
    // used by unrelated parallel tests.
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

fn stationary_bubble(offset: f64) -> (BubbleParameters, EnergyBalanceCalculator, BubbleState) {
    let _arrange = tracing::info_span!("arrange", offset).entered();
    let params = BubbleParameters::default();
    let calculator = EnergyBalanceCalculator::with_options(&params, false, false, false);
    let mut state = BubbleState::new(&params);
    state.temperature = calculator.ambient_temperature.in_unit::<Kelvin>() + offset;
    state.update_max_temperature();
    (params, calculator, state)
}

fn assert_roundoff(actual: f64, expected: f64) {
    // Fewer than 32 rounded operations occur in either the SI evaluation or
    // the independent spherical-wall/Euler oracle. No discretization error
    // enters an instantaneous rate or a single explicit Euler step.
    let gamma = 32.0 * f64::EPSILON / (1.0 - 32.0 * f64::EPSILON);
    let bound = gamma * expected.abs();
    let _assert = tracing::info_span!("assert", actual, expected, bound).entered();
    assert!(
        (actual - expected).abs() <= bound,
        "actual={actual}, expected={expected}, bound={bound}"
    );
}

#[test]
fn outward_conduction_reduces_internal_energy_and_inward_conduction_increases_it() {
    let _diagnostics = diagnostics();
    let _test = tracing::info_span!("outward_conduction_first_law").entered();
    for offset in [-32.0, 0.0, 32.0] {
        let (params, calculator, state) = stationary_bubble(offset);
        // At rest Pe=0 and Nu=2, so h A = (2 k/r)(4 pi r^2) = 8 pi k r.
        let outgoing = 8.0 * PI * params.thermal_conductivity * state.radius * offset;
        let heat = calculator.calculate_heat_transfer_rate(&state, 0.0);
        assert_roundoff(heat.in_unit::<Watt>(), outgoing);
        let rate = tracing::info_span!("act", caller = "calculate_energy_rate").in_scope(|| {
            calculator.calculate_energy_rate(
                &state,
                Pressure::from_unit::<Pascal>(params.p0),
                heat,
                Power::from_unit::<Watt>(0.0),
            )
        });
        // No wall work, phase change, chemistry, plasma or radiation: dU/dt=-Q_out.
        assert_roundoff(rate.in_unit::<Watt>(), -outgoing);
    }
}

#[test]
fn complete_energy_rate_has_the_physical_conduction_sign() {
    let _diagnostics = diagnostics();
    let _test = tracing::info_span!("complete_energy_rate_first_law").entered();
    for offset in [-32.0, 0.0, 32.0] {
        let (params, calculator, state) = stationary_bubble(offset);
        let diffusivity =
            params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);
        let actual = tracing::info_span!("act", caller = "calculate_complete_energy_rate")
            .in_scope(|| {
                calculator.calculate_complete_energy_rate(&state, params.p0, 0.0, diffusivity)
            });
        let expected = -8.0 * PI * params.thermal_conductivity * state.radius * offset;
        assert_roundoff(actual.in_unit::<Watt>(), expected);
    }
}

#[test]
fn temperature_update_cools_a_hot_bubble_and_warms_a_cold_bubble() {
    let _diagnostics = diagnostics();
    let _test = tracing::info_span!("temperature_update_first_law").entered();
    for offset in [-32.0, 32.0] {
        let (params, calculator, mut state) = stationary_bubble(offset);
        let ambient = calculator.ambient_temperature.in_unit::<Kelvin>();
        let initial = state.temperature;
        let peak = state.max_temperature;
        let cv =
            GAS_CONSTANT / state.gas_species.molecular_weight() / (state.gas_species.gamma() - 1.0);
        let capacity = state.mass() * cv;
        let conductance = 8.0 * PI * params.thermal_conductivity * state.radius;
        // dt=tau/16: the one-step Newton cooling excess is exactly 15/16 of
        // the original 32 K. The existing ambient floor still applies to
        // sub-ambient states; the rate test above observes their sign before it.
        let dt = capacity / conductance / 16.0;
        let expected = (ambient + offset * 15.0 / 16.0).max(ambient);
        tracing::info_span!("act", caller = "update_temperature_energy_balance").in_scope(|| {
            update_temperature_energy_balance(&calculator, &mut state, &params, params.p0, 0.0, dt)
        });
        assert_roundoff(state.temperature, expected);
        if offset > 0.0 {
            assert!(ambient < state.temperature && state.temperature < initial);
            assert_eq!(state.max_temperature.to_bits(), peak.to_bits());
        } else {
            assert!(initial < state.temperature);
        }
    }
}

#[test]
fn disabled_thermal_update_preserves_temperature_and_history() {
    let _diagnostics = diagnostics();
    let _test = tracing::info_span!("disabled_temperature_update").entered();
    let (mut params, calculator, mut state) = stationary_bubble(32.0);
    params.use_thermal_effects = false;
    let initial = state.temperature;
    let peak = state.max_temperature;
    tracing::info_span!("act").in_scope(|| {
        update_temperature_energy_balance(&calculator, &mut state, &params, params.p0, 0.0, 1.0)
    });
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(state.temperature.to_bits(), initial.to_bits());
    assert_eq!(state.max_temperature.to_bits(), peak.to_bits());
}

#[test]
fn compression_work_adds_energy_and_expansion_work_removes_it() {
    let _diagnostics = diagnostics();
    let _test = tracing::info_span!("pressure_work_first_law").entered();
    for velocity in [-1.0, 1.0] {
        let (params, calculator, mut state) = stationary_bubble(0.0);
        state.wall_velocity = velocity;
        let rate = tracing::info_span!("act").in_scope(|| {
            calculator.calculate_energy_rate(
                &state,
                Pressure::from_unit::<Pascal>(params.p0),
                Power::from_unit::<Watt>(0.0),
                Power::from_unit::<Watt>(0.0),
            )
        });
        let expected = -params.p0 * 4.0 * PI * state.radius.powi(2) * velocity;
        assert_roundoff(rate.in_unit::<Watt>(), expected);
    }
}
