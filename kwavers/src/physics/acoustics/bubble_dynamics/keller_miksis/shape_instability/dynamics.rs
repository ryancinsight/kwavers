use super::{ShapeModeState, N_MODES};

/// Advance shape mode amplitudes by one symplectic Euler step.
///
/// For each mode n:
///
/// ```text
/// a_ddot_n = G_n a_n - D_n a_dot_n
/// a_dot_n <- a_dot_n + a_ddot_n dt
/// a_n     <- a_n     + a_dot_n dt
/// ```
///
/// The velocity-first order is symplectic for the undamped capillary
/// oscillator and dissipative when viscosity contributes positive damping.
pub fn advance_shape_modes(
    modes: &mut ShapeModeState,
    r: f64,
    r_dot: f64,
    r_ddot: f64,
    sigma: f64,
    rho_l: f64,
    nu: f64,
    dt: f64,
) {
    if r < 1e-15 {
        return;
    }

    let r_inv = 1.0 / r;
    let r_dot_over_r = r_dot * r_inv;
    let r_ddot_over_r = r_ddot * r_inv;
    let r_dot_sq_over_r_sq = r_dot_over_r * r_dot_over_r;
    let r2_inv = r_inv * r_inv;
    let r3_inv = r2_inv * r_inv;

    for k in 0..N_MODES {
        let n = (k + 2) as f64;
        let damping = damping_coefficient(n, r_dot_over_r, r2_inv, nu);
        let driving = driving_term(n, r_ddot_over_r, r_dot_sq_over_r_sq, sigma, rho_l, r3_inv);

        let acceleration = driving * modes.amplitude[k] - damping * modes.rate[k];
        let rate = modes.rate[k] + acceleration * dt;
        modes.rate[k] = rate;
        modes.amplitude[k] += rate * dt;
    }
}

fn damping_coefficient(n: f64, r_dot_over_r: f64, r2_inv: f64, nu: f64) -> f64 {
    let viscous = 4.0 * nu * (n + 2.0) * (2.0 * n + 1.0) * r2_inv;
    3.0 * r_dot_over_r + viscous
}

fn driving_term(
    n: f64,
    r_ddot_over_r: f64,
    r_dot_sq_over_r_sq: f64,
    sigma: f64,
    rho_l: f64,
    r3_inv: f64,
) -> f64 {
    let inertial = (n - 1.0) * (r_ddot_over_r - (n + 2.0) * r_dot_sq_over_r_sq);
    let capillary = n * (n - 1.0) * (n + 2.0) * sigma * r3_inv / rho_l;
    inertial - capillary
}
