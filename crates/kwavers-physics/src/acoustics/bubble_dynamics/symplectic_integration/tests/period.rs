//! Test A — Minnaert period accuracy (Störmer-Verlet, O(h²) error).

use super::super::{BubbleSymplecticIntegrator, SymplecticConfig};
use super::helpers::{make_model, make_params};
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use kwavers_core::constants::numerical::TWO_PI;
use std::sync::Arc;

/// **Test A — Minnaert period accuracy (Störmer-Verlet, O(h²) error).**
///
/// Setup: R₀ = 10 µm, 5% radius perturbation, no driving.
///
/// Minnaert angular frequency (linearised RP):
///   ω₀ = (1/R₀) √((3γp_eq − 2σ/R₀) / ρ_L)
///
/// Validation: measured period vs. Minnaert period, error < 0.5%.
///
/// Reference: Minnaert, M. (1933). *Philos. Mag.* **16**, 235–248.
/// Error bound: Hairer et al. (2006) §I.3.1 — SV period error ~ (hω₀)²/24
/// = (2π/200)²/24 ≈ 0.04% ≪ 0.5%.
/// # Panics
/// - Panics if `SV step must not fail`.
///
#[test]
fn test_minnaert_period() {
    let r0 = 10e-6; // 10 µm
    let params = make_params(r0);
    let model = Arc::new(make_model(r0));

    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let pv = params.pv;
    let gamma = crate::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();

    // Minnaert angular frequency with corrected gas pressure closure:
    //   p_gas = (p0 + 2σ/R0 − pv)·(R0/R)^{3γ} + pv
    // Linearising around R = R0 gives spring constant 3γ(p0 + 2σ/R0 − pv),
    // so ω₀² = [3γ(p0 + 2σ/R0 − pv) − 2σ/R0] / (ρ_L R0²)
    let p_eq_gas = p0 + 2.0 * sigma / r0 - pv;
    let omega0 = ((3.0 * gamma * p_eq_gas - 2.0 * sigma / r0) / (rho_l * r0 * r0)).sqrt();
    let t_period = TWO_PI / omega0;
    let dt = t_period / 200.0; // 200 steps per period → (hω₀)²/24 ≈ 0.04% error

    let config = SymplecticConfig {
        dt,
        max_mach: 0.95,
        r_min_fraction: 0.01,
    };
    let integrator = BubbleSymplecticIntegrator::new(model.clone(), config);

    // 5% radius perturbation, zero velocity (linear regime)
    let mut state = BubbleState::at_equilibrium(&params);
    state.radius = 1.05 * r0;
    state.wall_velocity = 0.0;

    // Integrate for 100 periods; find the first return time (V=0, decelerating)
    let n_periods = 100;
    let total_steps = n_periods * 200;
    let mut t = 0.0;
    let mut periods: Vec<f64> = Vec::new();
    let mut prev_v = state.wall_velocity;
    let mut t_last_zero = 0.0;
    let mut first_zero_found = false;

    for _ in 0..total_steps {
        let v_before = state.wall_velocity;
        integrator
            .sv_step(&mut state, 0.0, 0.0, t)
            .expect("SV step must not fail");
        let v_after = state.wall_velocity;
        t += dt;

        // Detect V changing sign from negative to positive (minimum of R → new period)
        if prev_v < 0.0 && v_before < 0.0 && v_after >= 0.0 {
            if first_zero_found {
                periods.push(t - t_last_zero);
            }
            t_last_zero = t;
            first_zero_found = true;
        }
        prev_v = v_after;
    }

    assert!(
        !periods.is_empty(),
        "No complete oscillation periods detected in {} steps",
        total_steps
    );

    let period_meas = periods.iter().sum::<f64>() / periods.len() as f64;
    let rel_err = (period_meas - t_period).abs() / t_period;

    println!(
        "Minnaert period: measured {:.4} µs, exact {:.4} µs, error {:.4}%",
        period_meas * 1e6,
        t_period * 1e6,
        100.0 * rel_err
    );

    assert!(
        rel_err < 0.005,
        "Minnaert period error {:.3}% exceeds 0.5% (measured {:.4e} s, exact {:.4e} s)",
        100.0 * rel_err,
        period_meas,
        t_period
    );
}
