use crate::physics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};

/// Update thermodynamic state using adiabatic compression
///
/// T = T₀ (R₀/R)^{3(γ-1)},  P = P₀ (R₀/R)^{3γ}
pub(crate) fn update_thermodynamics(state: &mut BubbleState, bubble_params: &BubbleParameters) {
    if state.radius <= 0.0 {
        state.radius = 1e-9;
    }

    let gamma = bubble_params.gamma;
    let adiabatic_exponent = 3.0 * (gamma - 1.0);
    let radius_ratio = bubble_params.r0 / state.radius;
    state.temperature = bubble_params.t0 * radius_ratio.powf(adiabatic_exponent);

    let compression_ratio = radius_ratio.powi(3);
    state.pressure_internal = bubble_params.initial_gas_pressure * compression_ratio.powf(gamma);
}
