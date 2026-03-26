use super::thermodynamics::update_thermodynamics;
use crate::physics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use approx::assert_relative_eq;

#[test]
fn test_update_thermodynamics() {
    let params = BubbleParameters {
        r0: 5e-6,
        gamma: 1.4, // Diatomic gas like air
        t0: 300.0,
        initial_gas_pressure: 101325.0,
        ..Default::default()
    };

    let mut state = BubbleState::new(&params);

    // Initial state should match equilibrium
    update_thermodynamics(&mut state, &params);
    assert_relative_eq!(state.temperature, 300.0, epsilon = 1e-6);
    assert_relative_eq!(state.pressure_internal, 101325.0, epsilon = 1e-6);

    // Compress bubble to half radius
    state.radius = 2.5e-6; // R = R0 / 2
    update_thermodynamics(&mut state, &params);

    // Radius ratio = 2
    // Compression ratio = 8
    // T = T0 * 2^(0.4 * 3) = T0 * 2^1.2 ≈ 2.297 * 300
    let expected_t = 300.0 * 2.0_f64.powf(1.2);
    // P = P0 * 8^1.4 ≈ 18.379 * 101325
    let expected_p = 101325.0 * 8.0_f64.powf(1.4);

    assert_relative_eq!(state.temperature, expected_t, epsilon = 1e-2);
    assert_relative_eq!(state.pressure_internal, expected_p, epsilon = 1e-2);
}
