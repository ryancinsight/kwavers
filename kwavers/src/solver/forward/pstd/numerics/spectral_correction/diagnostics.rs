#[must_use]
pub fn compute_numerical_phase_velocity(k: f64, dx: f64, dt: f64, c_ref: f64) -> f64 {
    let k_mod = 2.0 * (k * dx / 2.0).sin() / dx;
    let arg = c_ref * dt * k_mod / 2.0;

    if arg < 1.0 {
        let omega_num = 2.0 * arg.asin() / dt;
        omega_num / k
    } else {
        0.0
    }
}

#[must_use]
pub fn compute_dispersion_error(k: f64, dx: f64, dt: f64, c_ref: f64) -> f64 {
    let c_num = compute_numerical_phase_velocity(k, dx, dt, c_ref);
    (c_num - c_ref).abs() / c_ref
}
