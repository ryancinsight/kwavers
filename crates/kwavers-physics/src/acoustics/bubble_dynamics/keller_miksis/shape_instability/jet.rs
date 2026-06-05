use super::JET_STANDOFF_CRITICAL;

/// Evaluate jet speed for a bubble collapsing near a rigid wall.
///
/// The Blake-Taib-Doherty stand-off model predicts jet formation when
/// `gamma = h / R_max <= 2`. For `gamma > 0.5`,
/// `V_jet = sqrt(2(p_inf - p_v)/rho_l) / (gamma - 0.5)`, capped at the liquid
/// sound speed.
#[must_use]
pub fn jet_speed(stand_off: f64, p_inf: f64, p_v: f64, rho_l: f64, c_l: f64) -> Option<f64> {
    if stand_off > JET_STANDOFF_CRITICAL {
        return None;
    }

    let delta_p = (p_inf - p_v).max(0.0);
    let rayleigh_speed = (2.0 * delta_p / rho_l.max(1.0)).sqrt();
    let denominator = (stand_off - 0.5).max(1e-3);
    Some((rayleigh_speed / denominator).min(c_l))
}
