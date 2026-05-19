use super::*;
use crate::core::constants::ATMOSPHERIC_PRESSURE;

#[test]
fn test_gilmore_initialization() {
    let params = BubbleParameters::default();
    let solver = GilmoreSolver::new(params.clone());

    assert!(solver.tait_b > 0.0);
    assert!(solver.tait_n > 1.0);
    // Enthalpy at ambient pressure must be positive (Tait EOS guarantee for p0 > 0).
    let h_ambient = GilmoreSolver::calculate_enthalpy(
        params.p0,
        params.rho_liquid,
        solver.tait_b,
        solver.tait_n,
    );
    assert!(h_ambient > 0.0, "h_ambient = {h_ambient}");
}

#[test]
fn test_sound_speed_calculation() {
    let params = BubbleParameters::default();
    let solver = GilmoreSolver::new(params);

    // At atmospheric pressure, sound speed should be close to 1500 m/s
    let c = solver.calculate_sound_speed(ATMOSPHERIC_PRESSURE);
    assert!(c > 1400.0 && c < 1600.0);

    // At higher pressure, sound speed increases
    let c_high = solver.calculate_sound_speed(10.0 * ATMOSPHERIC_PRESSURE);
    assert!(c_high > c);
}

/// Theorem: a bubble initialized at the correct mechanical equilibrium
/// (Young-Laplace: p_gas = p0 + 2σ/R0, no acoustic driving) must remain
/// stable under one RK4 step.
///
/// With the corrected Gilmore gas pressure closure
///   p_gas = (p0 + 2σ/R0 − pv)·(R0/R)^{3γ} + pv
/// the bubble-wall pressure at R = R0 satisfies:
///   p_wall = p_gas − 2σ/R0 = p0
/// which matches the far-field p_inf = p0, giving H = 0 and R̈ ≈ 0.
/// The radius after one step must therefore remain within 0.1% of R0.
/// # Panics
/// - Panics if post-step radius deviates by more than 0.1% from R0.
#[test]
fn step_rk4_bubble_stable_at_equilibrium() {
    let params = BubbleParameters::default();
    let solver = GilmoreSolver::new(params.clone());
    let state = BubbleState::at_equilibrium(&params);
    let r0 = state.radius;

    let dt = 1e-8; // 10 ns step
    let out = solver.step_rk4(&state, 0.0, 0.0, dt);

    assert!(
        out.radius > 0.0,
        "radius must stay positive after RK4 step; got {:.3e} m",
        out.radius
    );
    // Radius must not drift more than 0.1% from equilibrium
    let deviation = (out.radius - r0).abs() / r0;
    assert!(
        deviation < 1e-3,
        "bubble must remain near equilibrium; R0 = {r0:.3e} m, R1 = {:.3e} m (deviation {:.2e})",
        out.radius,
        deviation
    );
}

/// Theorem: under a compressive driving pressure (p_acoustic > 0, t = T/4
/// so sin(ωt) = 1 → the instantaneous field value is already positive),
/// a bubble initially at rest must contract (Ṙ < 0) after one step.
///
/// Note: `step_rk4` receives the instantaneous acoustic field value
/// directly; `calculate_acceleration` internally applies `p * sin(ωt)`.
/// At t = 0, sin(0) = 0, so we must pass t = T/4 to get maximum
/// compressive forcing.
/// # Panics
/// - Panics if assertion fails: `radius must remain positive; got {}`.
/// - Panics if assertion fails: `compressive driving must yield non-positive wall velocity; got {:.4e}`.
///
#[test]
fn step_rk4_compressive_forcing_contracts_bubble() {
    let params = BubbleParameters::default();
    let solver = GilmoreSolver::new(params.clone());
    let state = BubbleState::new(&params);

    let omega = 2.0 * std::f64::consts::PI * params.driving_frequency;
    let t_quarter = 0.25 / params.driving_frequency; // t where sin(ωt) = 1
    let p_acoustic = 200_000.0_f64; // 200 kPa — well above ambient
    let dt = 1e-8;

    let out = solver.step_rk4(&state, p_acoustic, t_quarter, dt);
    assert!(
        out.radius > 0.0,
        "radius must remain positive; got {}",
        out.radius
    );
    // At t_quarter, sin(ωt) = 1 → p_inf = p₀ + p_a > p₀.
    // The increased external pressure must compress the bubble: Ṙ < 0.
    assert!(
        out.wall_velocity <= 0.0,
        "compressive driving must yield non-positive wall velocity; got {:.4e}",
        out.wall_velocity
    );
    let _ = omega; // used indirectly via t_quarter
}

/// Theorem: `estimate_enthalpy_derivative` must genuinely depend on state.
///
/// Two `BubbleState` instances that differ only in `wall_acceleration` (R̈)
/// must produce different `dH/dt` values because the viscous and polytropic
/// contributions to `dp_wall/dt` include the `−4μR̈/R` term.
///
/// Also verifies the analytical form at zero wall-velocity (U = 0):
///   dp_wall/dt = −4μR̈/R  (viscous term only; gas and surface terms vanish with U=0)
///   dH_wall/dt = −4μR̈ / (R · ρ_wall)
///   dH_inf/dt  = p_a · ω · cos(ωt) / ρ_inf
///   dH/dt      = dH_wall_dt − dH_inf_dt
/// # Panics
/// - Panics if `acceleration A`.
/// - Panics if `acceleration B`.
///
#[test]
fn enthalpy_derivative_uses_state_wall_acceleration() {
    let params = BubbleParameters::default();
    let solver = GilmoreSolver::new(params.clone());

    let mut state_a = BubbleState::new(&params);
    state_a.wall_velocity = 0.0; // U = 0 isolates the R̈ term
    state_a.wall_acceleration = 0.0;

    let mut state_b = state_a.clone();
    state_b.wall_acceleration = 1.0e6; // non-zero R̈ affects dp_wall/dt

    let p_acoustic = 1.0e5; // 100 kPa driving
    let omega = 2.0 * std::f64::consts::PI * params.driving_frequency;
    let t = 0.0;

    // Need access to the private method through calculate_acceleration indirectly.
    // Use calculate_acceleration which calls estimate_enthalpy_derivative internally.
    // Two states with different R̈ but identical (R, U, gas) must give different R̈_out.
    let r_ddot_a = solver
        .calculate_acceleration(&state_a, p_acoustic, t)
        .expect("acceleration A");
    let r_ddot_b = solver
        .calculate_acceleration(&state_b, p_acoustic, t)
        .expect("acceleration B");

    // The Gilmore RHS includes (1 − U/C)·R/C · dH/dt; with U = 0:
    // R̈ = [(1+0)·H + (1−0)·R/C · dH/dt − 3/2·U²] / [R·(1−0)]
    //    = [H + R/C · dH/dt] / R
    // Two different R̈-dependent dH/dt must produce different output R̈_out.
    assert_ne!(
        r_ddot_a, r_ddot_b,
        "estimate_enthalpy_derivative must depend on state.wall_acceleration; \
         got identical accelerations {:.6e} with R̈=0 and R̈=1e6",
        r_ddot_a
    );

    // ── Analytical check at U=0, t=0 (cos(ω·0) = 1) ───────────────────
    // Corrected gas pressure closure: p_gas = (p0 + 2σ/R0 − pv)·(R0/R)^{3γ} + pv
    // dp_gas/dt    = 0  (U = 0)
    // dp_surface/dt = 0  (U = 0)
    // dp_viscous/dt = −4μ·R̈_a / R  = 0   (R̈_a = 0)
    // dp_inf/dt    = p_a · ω
    // dH_inf/dt    = p_a · ω / ρ_inf
    // → dH/dt_a    = 0 − p_a · ω / ρ_inf
    let p_inf = params.p0;
    let rho_inf = params.rho_liquid;
    let b = solver.tait_b;
    let n = solver.tait_n;
    let gamma = state_a.gas_species.gamma();
    let r = state_a.radius;
    let p_eq = params.p0 + 2.0 * params.sigma / params.r0 - params.pv;
    let p_gas = p_eq * (params.r0 / r).powf(3.0 * gamma) + params.pv;
    let p_wall_a = p_gas - 2.0 * params.sigma / r; // U=0 → viscous term zero
    let rho_wall_a = rho_inf * ((p_wall_a + b) / (p_inf + b)).powf(1.0 / n);
    let _ = rho_wall_a; // used to verify no panic on non-trivial density
    let expected_dh_dt_a = -(p_acoustic * omega) / rho_inf;

    // The acceleration from Gilmore at U=0 is: R̈ = [H + R/C·dH/dt] / R
    let h_wall_a = GilmoreSolver::calculate_enthalpy(p_wall_a, rho_inf, b, n);
    let h_inf_a = GilmoreSolver::calculate_enthalpy(p_inf, rho_inf, b, n);
    let h_a = h_wall_a - h_inf_a;
    let c_a = solver.calculate_sound_speed(p_wall_a);
    let expected_r_ddot_a = (h_a + r / c_a * expected_dh_dt_a) / r;

    let rel_err = (r_ddot_a - expected_r_ddot_a).abs() / expected_r_ddot_a.abs().max(1.0);
    assert!(
        rel_err < 1e-10,
        "Analytical Gilmore acceleration mismatch: expected {:.6e}, got {:.6e} \
         (relative error {:.2e})",
        expected_r_ddot_a,
        r_ddot_a,
        rel_err
    );
}

#[test]
fn test_gilmore_vs_keller_miksis_threshold() {
    let params = BubbleParameters::default();
    let solver = GilmoreSolver::new(params.clone());
    let mut state = BubbleState::new(&params);

    // Low amplitude - shouldn't need Gilmore
    state.wall_velocity = 10.0; // m/s
    assert!(!solver.should_use_gilmore(&state));

    // High amplitude - should use Gilmore
    state.wall_velocity = 200.0; // m/s (Mach > 0.1)
    assert!(solver.should_use_gilmore(&state));
}
