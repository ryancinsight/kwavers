//! Louisnard & Gomez (2003) Fig. 4 benchmark through the library's
//! own Keller–Miksis model and the implicit SDIRK2 integrator
//! (KW-PHYS-009 item 5 acceptance).
//!
//! This integration test is the executable acceptance vehicle for
//! the Build 22 Part 3 fix. It lives outside the lib-test target
//! because that target does not compile as found (an unrelated,
//! incomplete `growth/` workstream: `growth.rs` declares a
//! `#[cfg(test)] mod tests;` whose file does not exist, and
//! `growth/saturation/tests.rs` imports `tracing` /
//! `tracing_subscriber`, which are not dependencies of
//! `kwavers-physics`). The same benchmark also exists as unit tests
//! in `acoustics::mechanics::cavitation::core::louisnard_gomez`;
//! this file exercises the identical library path through the
//! public API so the acceptance can run today.
//!
//! Run with:
//! `cargo test --release -p kwavers-physics --test louisnard_gomez_fig4_benchmark -- --ignored`

use aequitas::systems::si::quantities::PressureRate;
use kwavers_physics::acoustics::bubble_dynamics::{
    AdaptiveBubbleConfig, BubbleParameters, BubbleState, GasSpecies, ImplicitBubbleIntegrator,
    InstantaneousPressure, KellerMiksisModel,
};
use kwavers_physics::acoustics::mechanics::cavitation::core::louisnard_gomez::{
    louisnard_gomez_growth_rate, LouisnardGomezParams,
};

const P0: f64 = 101_325.0;
const FREQUENCY: f64 = 26_500.0;

fn paper_params() -> LouisnardGomezParams {
    LouisnardGomezParams {
        diffusivity: 2.0e-9,
        saturation_concentration: 0.83,
        temperature: 293.0,
        ambient_pressure: P0,
        surface_tension: 0.073,
    }
}

/// Fig. 4 closure (prototype's validated implementation): isothermal
/// gas (κ = 1), vapour neglected, ρ = 1000, c = 1481, σ = 0.073,
/// μ = 0.001.
fn fig4_params(r0: f64) -> BubbleParameters {
    BubbleParameters {
        r0,
        p0: P0,
        rho_liquid: 1000.0,
        c_liquid: 1481.0,
        mu_liquid: 0.001,
        sigma: 0.073,
        pv: 0.0,
        gas_species: GasSpecies::Custom {
            gamma: 1.0,
            molecular_weight: 0.029,
        },
        use_thermal_effects: false,
        use_mass_transfer: false,
        driving_frequency: FREQUENCY,
        ..Default::default()
    }
}

fn fig4_config(period: f64) -> AdaptiveBubbleConfig {
    AdaptiveBubbleConfig {
        dt_max: period,
        // Step floor below the collapse spike's required step
        // (LSODA needed 1.6e-17 s at looser tolerances; a step
        // starting at the turnaround needs ~1e-19 s here).
        dt_min: 1e-20,
        rtol: 1e-8,
        atol: 1e-12,
        max_substeps: 100_000,
        ..AdaptiveBubbleConfig::default()
    }
}

fn fig4_grid_step(
    integrator: &mut ImplicitBubbleIntegrator<'_, KellerMiksisModel>,
    state: &mut BubbleState,
    p_ac: f64,
    omega: f64,
    grid_dt: f64,
    t: f64,
) {
    let p = InstantaneousPressure::from_pascals(-p_ac * (omega * t).sin());
    let dp_dt = PressureRate::from_base(-p_ac * omega * (omega * t).cos());
    integrator
        .integrate_adaptive(state, p, dp_dt, grid_dt, t)
        .expect("library KM + implicit SDIRK2 must traverse the Fig. 4 cycles");
}

/// Post-discard history window (cycles 5–60, 400 pts/cycle),
/// matching the prototype's evaluation protocol.
fn km_history(r0: f64, p_ac: f64) -> (Vec<f64>, Vec<f64>) {
    let n_cycles = 60;
    let discard = 5;
    let pts_per_cycle = 400;
    let period = 1.0 / FREQUENCY;
    let grid_dt = period / pts_per_cycle as f64;
    let total_pts = n_cycles * pts_per_cycle;

    let params = fig4_params(r0);
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    let mut integrator = ImplicitBubbleIntegrator::new(&model, fig4_config(period));
    let omega = 2.0 * std::f64::consts::PI * FREQUENCY;

    let mut grid_r = Vec::with_capacity(total_pts + 1);
    grid_r.push(state.radius);
    let mut t = 0.0;
    for _ in 0..total_pts {
        fig4_grid_step(&mut integrator, &mut state, p_ac, omega, grid_dt, t);
        t += grid_dt;
        grid_r.push(state.radius);
    }
    let start = discard * pts_per_cycle;
    let window_pts = total_pts - start;
    let radius = grid_r[start..=total_pts].to_vec();
    let time: Vec<f64> = (0..=window_pts).map(|i| i as f64 * grid_dt).collect();
    (radius, time)
}

/// Trajectory extremes over cycles 2..=n_cycles (fresh integrator
/// per cycle). LSODA reference, spike-resolved (solver internal
/// nodes and fs-scale dense-output windows around the turnaround;
/// two tolerance levels in exact agreement — physics review
/// 2026-11-27): at (5 µm, 5 bar) Rmin = 0.016 µm,
/// max |Ṙ| = 54,157 m/s; at (3 µm, 2 bar) Rmin = 0.0149 µm,
/// max |Ṙ| = 29,479 m/s. (Earlier reference readings from uniform
/// dense-output sampling — the diagnosis's 0.023 µm / 6044 m/s and
/// the Part 3 build's 0.018 µm / ≈1.5×10⁴ m/s at the 2-bar
/// anchors — missed the ~ps-wide spike apex.)
fn km_trajectory_extremes(r0: f64, p_ac: f64, n_cycles: usize) -> (f64, f64) {
    let period = 1.0 / FREQUENCY;
    let pts_per_cycle = 400;
    let grid_dt = period / pts_per_cycle as f64;

    let params = fig4_params(r0);
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    let omega = 2.0 * std::f64::consts::PI * FREQUENCY;

    let mut t = 0.0;
    let mut min_r = f64::INFINITY;
    let mut max_v: f64 = 0.0;
    for cycle in 0..n_cycles {
        let mut integrator = ImplicitBubbleIntegrator::new(&model, fig4_config(period));
        for _ in 0..pts_per_cycle {
            fig4_grid_step(&mut integrator, &mut state, p_ac, omega, grid_dt, t);
            t += grid_dt;
        }
        if cycle >= 1 {
            min_r = min_r.min(integrator.min_radius_seen());
            max_v = max_v.max(integrator.max_wall_speed_seen());
        }
    }
    (min_r, max_v)
}

/// Item-5 acceptance in miniature (always on): two Fig. 4 cycles at
/// (3 µm, 2 bar) integrate through collapse with the spike resolved
/// to the spike-resolved LSODA scale (Rmin = 0.0149 µm,
/// max |Ṙ| = 29,479 m/s; library-path measurement after the
/// +4μ/(ρc) LHS restoration, physics review 2026-11-27:
/// 0.0149 µm, 29,479 m/s — agreement to printed precision).
#[test]
fn fig4_collapse_cycles_complete() {
    let (min_r, max_v) = km_trajectory_extremes(3e-6, 200_000.0, 2);
    assert!(
        min_r > 0.00596e-6 && min_r < 0.03725e-6,
        "collapse spike must be resolved: min radius {min_r} m (LSODA 0.0149 µm, factor 2.5)"
    );
    assert!(
        max_v > 11_792.0 && max_v < 73_698.0,
        "collapse spike must be resolved: max |Ṙ| {max_v} m/s (LSODA 29,479 m/s, factor 2.5)"
    );
}

/// All six Fig. 4 anchors within the paper's factor-of-2 read bands,
/// plus the trajectory check at the strongest anchor.
#[test]
#[ignore = "expensive benchmark: six Keller-Miksis histories; run with -- --ignored"]
fn fig4_six_anchor_benchmark() {
    let params = paper_params();
    let anchors: [(f64, f64, f64); 6] = [
        (3e-6, 200_000.0, 200.0),
        (5e-6, 200_000.0, 100.0),
        (10e-6, 200_000.0, 30.0),
        (20e-6, 200_000.0, 10.0),
        (5e-6, 130_000.0, 30.0),
        (5e-6, 500_000.0, 300.0),
    ];
    let mut rates = Vec::new();
    for (r0, p_ac, anchor_um_s) in anchors {
        let (radius, time) = km_history(r0, p_ac);
        let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, r0, 1.0, &params) * 1e6;
        rates.push(rate_um_s);
        eprintln!("anchor R0 = {r0} m, p = {p_ac} Pa: {rate_um_s:.1} µm/s (paper ~{anchor_um_s})");
        assert!(
            rate_um_s > 0.5 * anchor_um_s && rate_um_s < 2.0 * anchor_um_s,
            "Fig. 4 anchor R0 = {r0} m, p = {p_ac} Pa: got {rate_um_s} µm/s, paper ~{anchor_um_s} µm/s"
        );
    }
    assert!(rates[0] > rates[1] && rates[1] > rates[2] && rates[2] > rates[3]);
    assert!(rates[4] < rates[1] && rates[1] < rates[5]);

    let (min_r, max_v) = km_trajectory_extremes(5e-6, 500_000.0, 3);
    eprintln!(
        "(5 µm, 5 bar) trajectory: Rmin = {min_r} m, max |Ṙ| = {max_v} m/s (LSODA 0.016 µm, 54,157 m/s, spike-resolved)"
    );
    assert!(
        min_r > 0.0064e-6 && min_r < 0.040e-6,
        "min radius {min_r} m must sit at the LSODA spike scale (0.016 µm, factor 2.5)"
    );
    assert!(
        max_v > 21_663.0 && max_v < 135_393.0,
        "max |Ṙ| {max_v} m/s must sit at the LSODA spike scale (54,157 m/s, factor 2.5)"
    );
}
