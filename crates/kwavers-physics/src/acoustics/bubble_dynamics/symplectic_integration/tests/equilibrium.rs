//! Test D — Exact equilibrium preservation.

use super::super::{BubbleSymplecticIntegrator, SymplecticConfig};
use super::helpers::{make_model, make_params};
use kwavers_core::constants::numerical::TWO_PI;
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use std::sync::Arc;

/// **Test D — Exact equilibrium preservation.**
///
/// At R = R₀, Ṙ = 0, no acoustic driving: K-M gives R̈ = 0.
/// Both half-kicks add zero → R and Ṙ remain exactly at equilibrium.
/// Assert |R − R₀|/R₀ < 1e-12 and |Ṙ| < 1e-12 after 1000 steps.
/// # Panics
/// - Panics if `SV step must not fail`.
///
#[test]
fn test_equilibrium_preserved() {
    let r0 = 10e-6;
    let params = make_params(r0);
    let model = Arc::new(make_model(r0));

    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;
    let omega0 = ((3.0 * gamma * p_eq - 2.0 * sigma / r0) / (rho_l * r0 * r0)).sqrt();
    let t_period = TWO_PI / omega0;
    let dt = t_period / 200.0;

    let config = SymplecticConfig {
        dt,
        ..Default::default()
    };
    let integrator = BubbleSymplecticIntegrator::new(model.clone(), config);

    // Exact mechanical equilibrium
    let mut state = BubbleState::at_equilibrium(&params);
    // at_equilibrium already sets radius = r0, wall_velocity = 0
    let mut t = 0.0;
    for _ in 0..1000 {
        integrator
            .sv_step(&mut state, 0.0, 0.0, t)
            .expect("SV step must not fail");
        t += dt;
    }

    let r_err = (state.radius - r0).abs() / r0;
    let v_err = state.wall_velocity.abs();
    println!(
        "Equilibrium preservation: |R−R₀|/R₀ = {:.2e}, |Ṙ| = {:.2e} m/s",
        r_err, v_err
    );

    assert!(
        r_err < 1e-12,
        "|R−R₀|/R₀ = {:.2e} exceeds 1e-12 — equilibrium drifted",
        r_err
    );
    assert!(
        v_err < 1e-12,
        "|Ṙ| = {:.2e} m/s exceeds 1e-12 — velocity drifted from zero",
        v_err
    );
}
