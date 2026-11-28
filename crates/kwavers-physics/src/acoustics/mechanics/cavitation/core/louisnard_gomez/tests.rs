use super::*;
use crate::acoustics::bubble_dynamics::{
    AdaptiveBubbleConfig, BubbleParameters, BubbleState, GasSpecies, ImplicitBubbleIntegrator,
    InstantaneousPressure, KellerMiksisModel,
};
use aequitas::systems::si::quantities::PressureRate;
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::error::{KwaversError, PhysicsError};

const P0: f64 = ATMOSPHERIC_PRESSURE;

fn paper_params() -> LouisnardGomezParams {
    // Louisnard & Gomez Fig. 4 conditions (air in water, 293 K).
    LouisnardGomezParams {
        diffusivity: 2.0e-9,
        saturation_concentration: 0.83,
        temperature: 293.0,
        ambient_pressure: P0,
        surface_tension: 0.073,
    }
}

#[test]
fn test_van_der_waals_at_equilibrium() {
    // At R = R0 the van der Waals law returns pg0 = p0 + 2σ/R0.
    let r0 = 5e-6;
    let sigma = 0.073;
    let pg = van_der_waals_gas_pressure(r0, r0, P0, sigma);
    assert_relative_eq!(pg, P0 + 2.0 * sigma / r0, epsilon = 1e-9);
}

#[test]
fn test_constant_history_exact_terms() {
    // R ≡ R0: R* ≡ 1, Rmax* = 1, q ≡ 1, so ⟨(u+1)^(4/3)⟩ = (u+1)^(4/3)
    // and I = (1/3)·∫₀^∞ (u+1)^(−4/3) du = (1/3)·3 = 1 exactly.
    // ⟨pg/p0⟩₄ = pg0/p0 = 1 + α_S.
    let r0 = 5e-6;
    let n = 200;
    let radius = vec![r0; n];
    let time: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
        .collect();
    let terms = cycle_average_terms(&radius, &time, r0, P0, 0.073);
    let alpha_s = 2.0 * 0.073 / (P0 * r0);
    assert_relative_eq!(terms.pg_over_p0_r4, 1.0 + alpha_s, epsilon = 1e-9);
    assert!(
        (terms.integral_i - 1.0).abs() < 0.01,
        "I for a constant history must be 1; got {}",
        terms.integral_i
    );
    assert_relative_eq!(terms.r_max_star, 1.0, epsilon = 1e-12);
}

#[test]
fn test_integral_i_against_direct_s_integration() {
    // Cross-check the u-substitution against a brute-force evaluation
    // of the defining integral I = ∫₀^∞ ds/⟨(3s + R*³)^(4/3)⟩ on a
    // synthetic expansion history (half-sine bump to R* = 6).
    let r0 = 5e-6;
    let n = 400;
    let radius: Vec<f64> = (0..n)
        .map(|i| {
            let phase = i as f64 / (n - 1) as f64;
            r0 * (1.0 + 5.0 * (std::f64::consts::PI * phase).sin())
        })
        .collect();
    let time: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
        .collect();
    let terms = cycle_average_terms(&radius, &time, r0, P0, 0.073);

    // Direct s-integration with the substitution-free definition.
    let r_star: Vec<f64> = radius.iter().map(|&r| r / r0).collect();
    let mut integral = 0.0;
    let s_max = 200.0;
    let n_s = 200_000;
    for i in 1..=n_s {
        let s_prev = s_max * (i - 1) as f64 / n_s as f64;
        let s = s_max * i as f64 / n_s as f64;
        let g = |sv: f64| {
            let vals: Vec<f64> = r_star
                .iter()
                .map(|&x| (3.0 * sv + x.powi(3)).powf(4.0 / 3.0))
                .collect();
            time_average(&vals, &time)
        };
        integral += 0.5 * (1.0 / g(s_prev) + 1.0 / g(s)) * (s - s_prev);
    }
    // Tail beyond s_max: ⟨(3s)^(4/3)⟩ → integrand ~ (3s)^(−4/3),
    // ∫ = (3)^(−4/3) · 3 · s_max^(−1/3).
    integral += 3.0_f64.powf(-4.0 / 3.0) * 3.0 * s_max.powf(-1.0 / 3.0);
    assert!(
        (terms.integral_i - integral).abs() / integral < 0.05,
        "u-substitution I = {} vs direct s-integration {integral}",
        terms.integral_i
    );
}

#[test]
fn test_regime_flags() {
    // Paper conditions: in regime.
    let r = louisnard_gomez_regime(5e-6, 200_000.0, 26_500.0, 1.0);
    assert!(r.in_validated_regime());
    // Formation-model burst conditions (1 MHz, 10 MPa): out of regime
    // in frequency and pressure, in regime in radius and saturation.
    let r = louisnard_gomez_regime(5e-6, 10_530_000.0, 1_000_000.0, 0.2);
    assert!(r.frequency_outside_validated);
    assert!(r.pressure_outside_validated);
    assert!(!r.radius_outside_validated);
    assert!(!r.saturation_outside_validated);
    assert!(!r.in_validated_regime());
}

#[test]
fn test_checked_rejects_burst_extrapolation() {
    let params = paper_params();
    let n = 100;
    let radius = vec![5e-6; n];
    let time: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
        .collect();
    // In regime the checked form is the unchecked law, bit for bit.
    let checked = louisnard_gomez_growth_rate_checked(
        &radius, &time, 5e-6, 1.0, 200_000.0, 26_500.0, &params,
    )
    .unwrap();
    let plain = louisnard_gomez_growth_rate(&radius, &time, 5e-6, 1.0, &params);
    assert_eq!(checked.to_bits(), plain.to_bits());
    // Burst extrapolation (1 MHz) is refused by the checked form. The
    // frequency bound is tested before the pressure bound, so it is
    // the parameter the error names.
    let burst = louisnard_gomez_growth_rate_checked(
        &radius,
        &time,
        5e-6,
        0.2,
        10_530_000.0,
        1_000_000.0,
        &params,
    );
    assert!(matches!(
        burst,
        Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            ref parameter,
            value,
            ..
        })) if parameter == "frequency" && value.to_bits() == 1_000_000.0_f64.to_bits()
    ));
}

#[test]
fn test_checked_rejects_nan_and_degenerate_params() {
    // Guard soundness (physics review 2026-11-27): a checked
    // variant must refuse NaN and degenerate medium parameters,
    // not return Ok(0.0) / Ok(NaN) — the unchecked law's silent
    // 0.0 is exactly what the checked form exists to prevent.
    let params = paper_params();
    let n = 100;
    let radius = vec![5e-6; n];
    let time: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 / 26_500.0)
        .collect();
    let call = |r0: f64, params: &LouisnardGomezParams| {
        louisnard_gomez_growth_rate_checked(&radius, &time, r0, 1.0, 200_000.0, 26_500.0, params)
    };
    assert!(call(f64::NAN, &params).is_err());
    let mut q = params;
    q.diffusivity = 0.0;
    assert!(call(5e-6, &q).is_err());
    q = params;
    q.diffusivity = f64::NAN;
    assert!(call(5e-6, &q).is_err());
    q = params;
    q.saturation_concentration = 0.0;
    assert!(call(5e-6, &q).is_err());
    q = params;
    q.temperature = -1.0;
    assert!(call(5e-6, &q).is_err());
    q = params;
    q.ambient_pressure = 0.0;
    assert!(call(5e-6, &q).is_err());
    // The regime classifier flags a NaN radius as outside.
    assert!(louisnard_gomez_regime(f64::NAN, 200_000.0, 26_500.0, 1.0).radius_outside_validated);
}

/// Keller–Miksis parameters at the Louisnard & Gomez Fig. 4
/// closure (the prototype's validated implementation,
/// formation/b6a_law.py): isothermal gas (κ = 1), vapour
/// neglected, ρ = 1000 kg/m³, c = 1481 m/s (the paper's value),
/// σ = 0.073 N/m, μ = 0.001 Pa·s.
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
        driving_frequency: 26_500.0,
        ..Default::default()
    }
}

/// Integrator configuration for the Fig. 4 histories: tolerances
/// an order tighter than the library defaults (the law reads the
/// expansion phase of a stiff trajectory; the prototype's LSODA
/// ground truth ran at rtol 1e-9), a step floor far below the
/// collapse spike's timescale (LSODA itself needed steps to
/// 1.6×10⁻¹⁷ s; a step starting at the turnaround needs ~10⁻¹⁹ s
/// at these tolerances, so the floor must not bind), and a
/// substep budget sized for sub-cycling through two collapses
/// per main step at the strongest anchor.
fn fig4_config(period: f64) -> AdaptiveBubbleConfig {
    AdaptiveBubbleConfig {
        dt_max: period,
        dt_min: 1e-20,
        rtol: 1e-8,
        atol: 1e-12,
        max_substeps: 100_000,
        ..AdaptiveBubbleConfig::default()
    }
}

/// Advance the library Keller–Miksis model by one output-grid
/// step (T/400) under the Fig. 4 tone, following the integrator's
/// zero-order-hold drive contract: instantaneous pressure
/// p(t) = −p_ac·sin(ωt) and dp/dt = −p_ac·ω·cos(ωt) sampled at
/// the step start.
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

/// Integrate the library Keller–Miksis model at the Louisnard &
/// Gomez Fig. 4 conditions and return the post-discard history
/// window (cycles 5–60, uniform 400-points-per-cycle grid),
/// matching the prototype's evaluation protocol (b6a_law.km_solve:
/// 60 cycles, first 5 discarded, the law evaluated over the whole
/// post-discard window — a last-cycle evaluation is poisoned at
/// the strongest anchor by individual afterbounce-heavy cycles).
///
/// Histories come from the library's own KM model driven by the
/// implicit SDIRK2 integrator (KW-PHYS-009 item 5): the previous
/// test-local explicit RK4 transcription is gone. The explicit
/// step-doubling family cannot traverse the collapse spike
/// without injecting energy (at (5 µm, 5 bar) it returned a
/// window rate of ~10⁷ µm/s against the paper's ~300 µm/s, and
/// it stalled outright at (3 µm, 2 bar)); the L-stable implicit
/// path damps the unresolved spike instead, and the law's R⁴
/// weighting reads the expansion phase it resolves.
fn km_history(r0: f64, p_ac: f64, frequency: f64) -> (Vec<f64>, Vec<f64>) {
    let n_cycles = 60;
    let discard = 5;
    let pts_per_cycle = 400;
    let period = 1.0 / frequency;
    let grid_dt = period / pts_per_cycle as f64;
    let total_pts = n_cycles * pts_per_cycle;

    let params = fig4_params(r0);
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    let mut integrator = ImplicitBubbleIntegrator::new(&model, fig4_config(period));
    let omega = 2.0 * std::f64::consts::PI * frequency;

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

/// Trajectory extremes (min radius, max |Ṙ|) over cycles
/// 2..=n_cycles at the Fig. 4 conditions, from a fresh implicit
/// integrator per cycle so the start-up cycle's extremes do not
/// pollute the window. The output grid undersamples the ~ns
/// collapse spike by construction, so the extremes come from the
/// integrator's accepted sub-steps. LSODA reference,
/// spike-resolved (solver internal nodes and fs-scale
/// dense-output windows around the turnaround; two tolerance
/// levels in exact agreement — physics review 2026-11-27): at
/// (5 µm, 5 bar) Rmin = 0.016 µm, max |Ṙ| = 54,157 m/s; at
/// (5 µm, 2 bar) Rmin = 0.0297 µm, max |Ṙ| = 20,842 m/s; at
/// (3 µm, 2 bar) Rmin = 0.0149 µm, max |Ṙ| = 29,479 m/s.
/// (Earlier reference readings — the B22 Part 3 diagnosis's
/// 0.023 µm / 6044 m/s and the Part 3 build's 0.018 µm /
/// ≈1.5×10⁴ m/s at the 2-bar anchors — were uniform-grid
/// samplings that missed the ~ps-wide spike apex; their Rmax
/// values were exact.)
fn km_trajectory_extremes(r0: f64, p_ac: f64, frequency: f64, n_cycles: usize) -> (f64, f64) {
    let period = 1.0 / frequency;
    let pts_per_cycle = 400;
    let grid_dt = period / pts_per_cycle as f64;

    let params = fig4_params(r0);
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(&params);
    let omega = 2.0 * std::f64::consts::PI * frequency;

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

#[test]
fn test_louisnard_gomez_fig4_anchor() {
    // Louisnard & Gomez (2003) Fig. 4 anchor: R0 = 5 µm,
    // p = 2 bar at 26.5 kHz in saturated water grows at
    // ~100 µm/s. The paper's anchors are order-of-magnitude
    // reads (factor ~2 per the prototype's validation notes);
    // the prototype's LSODA gave 94.6 µm/s for this anchor.
    // Assert the factor-of-2 band. The history is produced by
    // the library's own KM + implicit SDIRK2 integrator
    // (KW-PHYS-009 item 5); the six-anchor sweep is the ignored
    // benchmark below.
    let params = paper_params();
    let (radius, time) = km_history(5e-6, 200_000.0, 26_500.0);
    let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, 5e-6, 1.0, &params) * 1e6;
    assert!(
        rate_um_s > 50.0 && rate_um_s < 200.0,
        "Fig. 4 anchor R0 = 5 µm, p = 2 bar: got {rate_um_s} µm/s, paper ~100 µm/s"
    );
}

#[test]
#[ignore = "expensive benchmark: six Keller-Miksis histories; run with -- --ignored"]
fn test_louisnard_gomez_fig4_benchmark() {
    // Louisnard & Gomez (2003) Fig. 4 growth rates at 26.5 kHz,
    // saturated water — all six of the paper's anchors, with the
    // histories generated by the library's own KM model and the
    // implicit SDIRK2 integrator (KW-PHYS-009 item 5). Assert a
    // factor-of-2 band around each anchor and the correct
    // ordering in R0 and p. Prototype LSODA references:
    // (3, 2) 222.9, (5, 2) 94.6, (5, 5) 275.3 µm/s against the
    // paper's ~200 / ~100 / ~300 µm/s reads.
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
        let (radius, time) = km_history(r0, p_ac, 26_500.0);
        let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, r0, 1.0, &params) * 1e6;
        rates.push(rate_um_s);
        assert!(
            rate_um_s > 0.5 * anchor_um_s && rate_um_s < 2.0 * anchor_um_s,
            "Fig. 4 anchor R0 = {r0} m, p = {p_ac} Pa: got {rate_um_s} µm/s, paper ~{anchor_um_s} µm/s"
        );
    }
    // Ordering: at 2 bar the rate falls with R0; at R0 = 5 µm
    // the rate rises with p (1.3 bar < 2 bar < 5 bar).
    assert!(rates[0] > rates[1] && rates[1] > rates[2] && rates[2] > rates[3]);
    assert!(rates[4] < rates[1] && rates[1] < rates[5]);

    // Trajectory check at the strongest anchor: the accepted
    // sub-steps must resolve the collapse spike to the
    // spike-resolved LSODA reference's scale (Rmin = 0.016 µm,
    // max |Ṙ| = 54,157 m/s), within a factor 2.5 either way —
    // an integrator that skips or damps the collapse entirely,
    // or one whose rebound runs away, fails here even if a
    // growth-rate band elsewhere hid it. (Library measurement,
    // physics review 2026-11-27: Rmin = 0.0160 µm,
    // max |Ṙ| = 54,157 m/s — the LSODA value to printed
    // precision, after the +4μ/(ρc) LHS term was restored.)
    let (min_r, max_v) = km_trajectory_extremes(5e-6, 500_000.0, 26_500.0, 3);
    assert!(
        min_r > 0.0064e-6 && min_r < 0.040e-6,
        "Fig. 4 (5 µm, 5 bar) trajectory: min radius {min_r} m must sit at the LSODA spike scale (0.016 µm, factor 2.5)"
    );
    assert!(
        max_v > 21_663.0 && max_v < 135_393.0,
        "Fig. 4 (5 µm, 5 bar) trajectory: max |Ṙ| {max_v} m/s must sit at the LSODA spike scale (54,157 m/s, factor 2.5)"
    );
}
