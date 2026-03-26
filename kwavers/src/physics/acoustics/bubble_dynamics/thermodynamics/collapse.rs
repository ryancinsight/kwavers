//! Collapse temperature calculator

/// Calculate bubble collapse temperature using adiabatic heating
/// Theorem: T_final = T0 * (R0/R)^(3(γ-1))
/// Literature: Yasui (1995), Moss et al. (1997)
#[must_use]
pub fn calculate_collapse_temperature(
    bubble_params: &crate::physics::bubble_dynamics::BubbleParameters,
    collapse_ratio: f64,
) -> f64 {
    // Adiabatic heating during bubble collapse
    // For air: γ = 1.4, so T ∝ (R0/R)^(3*0.4) = (R0/R)^1.2
    let gamma = bubble_params.gamma;
    let adiabatic_exponent = 3.0 * (gamma - 1.0);

    bubble_params.t0 * (1.0 / collapse_ratio).powf(adiabatic_exponent)
}
