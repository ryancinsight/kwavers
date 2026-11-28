use super::*;
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

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

/// Keller–Miksis RHS transcribed from the prototype's validated
/// implementation (formation/b6a_law.py `km_rhs`): isothermal gas
/// (κ = 1), vapour neglected, no elasticity (G = 0).
///
/// A test-local integrator is used because the library's native
/// KM cannot currently produce histories in this regime: its
/// acceleration returns an error above wall Mach 0.95 and its
/// adaptive integrator's stability monitor rejects |Ṙ| > 0.1·c,
/// while the Fig. 4 collapses pass both bounds (the prototype's
/// LSODA integration, rtol 1e-9, handles them; the F–S law
/// suppresses the collapse phase by R⁴ weighting, so the law's
/// outputs are insensitive to collapse detail). Filed as
/// KW-PHYS-009.
fn km_rhs_test(t: f64, r: f64, v: f64, r0: f64, p_ac: f64, f: f64) -> (f64, f64) {
    const RHO: f64 = 1000.0;
    const C_SND: f64 = 1481.0; // Louisnard & Gomez value
    const SIGMA: f64 = 0.073;
    const MU: f64 = 0.001;
    const KAPPA: f64 = 1.0;
    let r = r.max(r0 * 1e-3);
    let om = 2.0 * std::f64::consts::PI * f;
    let p_ac_t = -p_ac * (om * t).sin();
    let p_ac_dot = -p_ac * om * (om * t).cos();
    let pg = (P0 + 2.0 * SIGMA / r0) * (r0 / r).powf(3.0 * KAPPA);
    let pw = pg - 2.0 * SIGMA / r - 4.0 * MU * v / r;
    let p_inf = P0 + p_ac_t;
    let a = pw - p_inf;
    let d_a = -3.0 * KAPPA * pg * v / r + 2.0 * SIGMA * v / (r * r) + 4.0 * MU * v * v / (r * r)
        - p_ac_dot;
    let coef = (1.0 - v / C_SND) * r + 4.0 * MU / (RHO * C_SND);
    let rhs = (1.0 + v / C_SND) * a / RHO + (r / (RHO * C_SND)) * d_a
        - 1.5 * (1.0 - v / (3.0 * C_SND)) * v * v;
    (v, rhs / coef)
}

fn rk4_step_test(t: f64, r: f64, v: f64, h: f64, r0: f64, p_ac: f64, f: f64) -> (f64, f64) {
    let (k1r, k1v) = km_rhs_test(t, r, v, r0, p_ac, f);
    let (k2r, k2v) = km_rhs_test(
        t + 0.5 * h,
        r + 0.5 * h * k1r,
        v + 0.5 * h * k1v,
        r0,
        p_ac,
        f,
    );
    let (k3r, k3v) = km_rhs_test(
        t + 0.5 * h,
        r + 0.5 * h * k2r,
        v + 0.5 * h * k2v,
        r0,
        p_ac,
        f,
    );
    let (k4r, k4v) = km_rhs_test(t + h, r + h * k3r, v + h * k3v, r0, p_ac, f);
    (
        r + h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r),
        v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )
}

/// Integrate the test-local Keller–Miksis at the Louisnard &
/// Gomez Fig. 4 conditions and return the post-discard
/// history window (cycles 5–60, uniform 400-points-per-cycle
/// grid), matching the prototype's evaluation protocol.
/// Controller (each element earned by a failure mode
/// against the prototype's LSODA ground truth): step-doubling
/// RK4 at rtol 1e-10, atol (1e-14, 1e-8) (10× tighter than
/// the prototype's LSODA tolerances — the explicit scheme
/// needs them at the strongest anchor); the half-step result accepted as-is (the
/// Richardson extrapolation injects energy in the stiff
/// rebound); growth capped at 1.5× (a 5× cap jumps over the
/// collapse onset); a relative-change rejection |ΔR| ≤ R/2
/// (an unbounded accepted step overshoots past Rmin into
/// unrecoverable R⁻³ stiffness); and a bounded ride-through
/// at the step floor for the stiff instant itself — the
/// explicit analogue of LSODA switching to BDF there (an
/// unbounded floor-accept is poison: it once let a rebound
/// step through that exploded the bubble to metres).
///
/// The two strongest anchors are excluded from the sweep
/// below: (3 µm, 2 bar), Rmax* = 33, stalls this driver in
/// the stiff rebound, and at (5 µm, 5 bar) the driver's
/// history is converged in tolerance yet departs
/// qualitatively from LSODA's (window rate ~10⁷ µm/s vs
/// 275 µm/s — the explicit scheme settles into a
/// non-collapsing mode the implicit scheme does not).
/// Both anchors' validation stands on the prototype's LSODA
/// numbers (222.9 and 275.3 µm/s vs the paper's ~200 and
/// ~300 µm/s reads). This is the same gap as KW-PHYS-009
/// item 5 for the library's own integrator.
fn km_history(r0: f64, p_ac: f64, frequency: f64) -> (Vec<f64>, Vec<f64>) {
    // Prototype protocol (b6a_law.km_solve): 60 cycles, the
    // first 5 discarded, and the law evaluated over the whole
    // post-discard window — NOT the last cycle alone. A
    // last-cycle evaluation is poisoned at the strongest
    // anchor (5 µm, 5 bar) by individual non-collapsing /
    // afterbounce-heavy cycles; the window average is what
    // the paper's Fig. 4 reports.
    let n_cycles = 60;
    let discard = 5;
    let pts_per_cycle = 400;
    let period = 1.0 / frequency;
    let t_end = n_cycles as f64 * period;
    let grid_dt = period / pts_per_cycle as f64;
    let total_pts = n_cycles * pts_per_cycle;
    let mut grid_r = Vec::with_capacity(total_pts + 1);
    let (mut r, mut v) = (r0, 0.0);
    let mut t = 0.0;
    grid_r.push(r);
    let mut h = grid_dt / 10.0;
    let mut next_grid = grid_dt;
    while t < t_end - 1e-15 {
        // Advance to the next grid point with step-doubling.
        let target = next_grid.min(t_end);
        let mut iterations = 0_u64;
        while t < target - 1e-15 {
            iterations += 1;
            assert!(
                iterations < 20_000_000,
                "KM test integrator stuck at t = {t:e}, h = {h:e}, R = {r:e}, V = {v:e}"
            );
            let hh = h.min(target - t);
            let (rf, vf) = rk4_step_test(t, r, v, hh, r0, p_ac, frequency);
            let (rh, vh) = rk4_step_test(t, r, v, hh / 2.0, r0, p_ac, frequency);
            let (rh, vh) = rk4_step_test(t + hh / 2.0, rh, vh, hh / 2.0, r0, p_ac, frequency);
            // A substage can go non-finite when the collapse
            // drives the wall through the Keller–Miksis
            // singularity (|Ṙ| → c makes the LHS coefficient
            // vanish). Reject and shrink — never let a NaN
            // reach the step-size factor (NaN comparisons are
            // false, which would otherwise grow h and loop
            // forever on the same interval).
            if !rh.is_finite() || !vh.is_finite() {
                h = hh * 0.25;
                continue;
            }
            // Step-doubling error estimate (un-divided
            // difference — the conservative acceptance test).
            // The half-step result is accepted as-is: the
            // Richardson-extrapolated value injects energy in
            // the stiff collapse rebound and runs the state
            // away (|Ṙ| → 10⁴ m/s at R → 50 nm).
            let err = ((rf - rh).abs() / (1e-14 + 1e-10 * rh.abs()))
                .max((vf - vh).abs() / (1e-8 + 1e-10 * vh.abs()));
            if !err.is_finite() {
                h = hh * 0.25;
                continue;
            }
            // Relative-change rejection: no accepted step may
            // move R by more than half itself. Without it a
            // step entering the collapse overshoots past the
            // true Rmin (~1 µm) to tens of nm, after which the
            // R⁻³ gas law is unrecoverable stiffness.
            let change_ok = (rh - r).abs() <= 0.5 * r;
            if err <= 1.0 && change_ok {
                r = rh;
                v = vh;
                t += hh;
            } else if hh <= 1e-13 && change_ok {
                // Bounded ride-through at the step floor: the
                // stiff instant (deepest collapse) cannot be
                // resolved by any explicit step, but a floor
                // step whose displacement is itself bounded
                // moves the state negligibly — accept it and
                // stay small, the explicit analogue of LSODA
                // switching to BDF there. An UNBOUNDED
                // floor-accept is poison (it let a rebound
                // step through that exploded the bubble), so
                // the change bound is the gate.
                r = rh;
                v = vh;
                t += hh;
                h = hh;
                continue;
            }
            // Growth capped at 1.5×: a 5× cap lets h jump
            // over the collapse onset between checks.
            let factor = if err > 0.0 {
                (0.9 * err.powf(-0.2)).clamp(0.2, 1.5)
            } else {
                1.5
            };
            h = (hh * factor).max(1e-13);
        }
        grid_r.push(r);
        next_grid += grid_dt;
    }
    let start = discard * pts_per_cycle;
    let window_pts = total_pts - start;
    let radius = grid_r[start..=total_pts].to_vec();
    let time: Vec<f64> = (0..=window_pts).map(|i| i as f64 * grid_dt).collect();
    (radius, time)
}

#[test]
fn test_louisnard_gomez_fig4_anchor() {
    // Louisnard & Gomez (2003) Fig. 4 anchor: R0 = 5 µm,
    // p = 2 bar at 26.5 kHz in saturated water grows at
    // ~100 µm/s. The paper's anchors are order-of-magnitude
    // reads (factor ~2 per the prototype's validation notes);
    // the prototype's LSODA gave 94.6 µm/s for this anchor.
    // Assert the factor-of-2 band. The four-anchor sweep is
    // the ignored benchmark below (KM histories are too
    // expensive for the always-on suite in debug builds);
    // two anchors are excluded from it — see km_history's
    // docs.
    let params = paper_params();
    let (radius, time) = km_history(5e-6, 200_000.0, 26_500.0);
    let rate_um_s = louisnard_gomez_growth_rate(&radius, &time, 5e-6, 1.0, &params) * 1e6;
    assert!(
        rate_um_s > 50.0 && rate_um_s < 200.0,
        "Fig. 4 anchor R0 = 5 µm, p = 2 bar: got {rate_um_s} µm/s, paper ~100 µm/s"
    );
}

#[test]
#[ignore = "expensive benchmark: four Keller-Miksis histories; run with -- --ignored"]
fn test_louisnard_gomez_fig4_benchmark() {
    // Louisnard & Gomez (2003) Fig. 4 growth rates at 26.5 kHz,
    // saturated water. Assert a factor-of-2 band around each
    // anchor and the correct ordering in R0 and p. Four of the
    // paper's six anchors; the two strongest are excluded
    // because their histories defeat the explicit test driver
    // (see km_history's docs): (3 µm, 2 bar) stalls it, and at
    // (5 µm, 5 bar) its converged history departs
    // qualitatively from LSODA's. Both are covered by the
    // prototype's LSODA integration (222.9 and 275.3 µm/s vs
    // the paper's ~200 and ~300 µm/s).
    let params = paper_params();
    let anchors: [(f64, f64, f64); 4] = [
        (5e-6, 200_000.0, 100.0),
        (10e-6, 200_000.0, 30.0),
        (20e-6, 200_000.0, 10.0),
        (5e-6, 130_000.0, 30.0),
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
    // the 2 bar rate exceeds the 1.3 bar rate.
    assert!(rates[0] > rates[1] && rates[1] > rates[2]);
    assert!(rates[0] > rates[3]);
}
