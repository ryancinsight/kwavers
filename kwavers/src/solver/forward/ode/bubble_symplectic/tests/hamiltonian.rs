//! Test B — Hamiltonian non-drift over 1000 periods (Störmer-Verlet).

use super::super::{BubbleSymplecticIntegrator, SymplecticConfig};
use super::helpers::{bubble_hamiltonian, make_model, make_params};
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;
use std::f64::consts::PI;
use std::sync::Arc;

/// **Test B — Hamiltonian non-drift over 1000 periods (Störmer-Verlet).**
///
/// For the Rayleigh-Plesset / Keller-Miksis mechanical sub-system, the true
/// Hamiltonian H = ½ρ_L R³Ṙ² + V_eff(R) is exactly conserved by the undamped
/// RP equation (dH/dt = 0).
///
/// The position-Verlet (SV) integrator applied to R̈ = f(R, Ṙ) with
/// velocity-dependent f is NOT equivalent to canonical SV on a separable
/// Hamiltonian; the force term −3Ṙ²/(2R) is O(V²/R) and introduces O(h)
/// energy oscillations (not O(h²) as for separable H = T(p) + V(q)).
///
/// Nevertheless, the energy DOES NOT drift monotonically:
///   - Dissipative system (μ ≠ 0): H → 0 within ~100 periods.
///   - This integrator: H oscillates, stays in [0.5 H₀, 2 H₀] over 1000 periods.
///
/// The key distinguishing property is bounded oscillation vs secular drift.
///
/// Assertion: H_min > 0.5 × H₀  and  H_max < 2.0 × H₀  over 1000 periods.
/// # Panics
/// - Panics if `SV step must not fail`.
///
#[test]
fn test_hamiltonian_no_drift() {
    let r0 = 10e-6;
    let params = make_params(r0);
    let model = Arc::new(make_model(r0));

    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;
    let omega0 = ((3.0 * gamma * p_eq - 2.0 * sigma / r0) / (rho_l * r0 * r0)).sqrt();
    let t_period = 2.0 * PI / omega0;
    let dt = t_period / 200.0;

    let config = SymplecticConfig {
        dt,
        max_mach: 0.95,
        r_min_fraction: 0.01,
    };
    let integrator = BubbleSymplecticIntegrator::new(model.clone(), config);

    let mut state = BubbleState::at_equilibrium(&params);
    state.radius = 1.05 * r0;
    state.wall_velocity = 0.0;

    let h0 = bubble_hamiltonian(state.radius, state.wall_velocity, &params, 1000);
    let mut h_min = h0;
    let mut h_max = h0;

    let total_steps = 1000 * 200; // 1000 periods
    let mut t = 0.0;

    for _ in 0..total_steps {
        integrator
            .sv_step(&mut state, 0.0, 0.0, t)
            .expect("SV step must not fail");
        t += dt;
        let h = bubble_hamiltonian(state.radius, state.wall_velocity, &params, 200);
        h_min = h_min.min(h);
        h_max = h_max.max(h);
    }

    let h_spread = (h_max - h_min) / h0.abs().max(1e-30);
    println!(
        "H over 1000 periods: min={:.4e}  max={:.4e}  H₀={:.4e}  spread={:.2}%",
        h_min,
        h_max,
        h0,
        100.0 * h_spread
    );

    // Primary check: no secular LOSS (energy stays above 50 % of H₀).
    // A viscous system (non-zero μ) would have H_min → 0 within a few hundred periods.
    assert!(
        h_min > 0.5 * h0,
        "H dropped below 50% of H₀ — secular energy loss detected \
         (H_min={:.4e}, H₀={:.4e})",
        h_min,
        h0
    );
    // Secondary check: no secular GAIN (energy stays below 200 % of H₀).
    assert!(
        h_max < 2.0 * h0,
        "H exceeded 200% of H₀ — energy is growing (H_max={:.4e}, H₀={:.4e})",
        h_max,
        h0
    );
}
