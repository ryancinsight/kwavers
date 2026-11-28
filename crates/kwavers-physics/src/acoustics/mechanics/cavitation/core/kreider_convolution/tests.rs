use super::*;

#[test]
fn test_saturation_pressure_anchors() {
    // Prototype self-test anchors (IAPWS Wagner).
    let p37 = saturation_vapour_pressure(310.15);
    assert!((p37 - 6280.0).abs() / 6280.0 < 0.02, "p_sat(37 C) = {p37}");
    let p100 = saturation_vapour_pressure(373.15);
    assert!(
        (p100 - 101_325.0).abs() / 101_325.0 < 0.02,
        "p_sat(100 C) = {p100}"
    );
    assert_eq!(saturation_vapour_pressure(WATER_T_CRIT_K), WATER_P_CRIT_PA);
}

#[test]
fn test_latent_heat_vanishes_at_critical() {
    assert_eq!(latent_heat_molar(WATER_T_CRIT_K), 0.0);
    assert!(latent_heat_molar(310.0) > 0.0);
}

#[test]
fn test_convolution_reduces_to_epstein_plesset() {
    // Fixed R, F = −C∞ (wall concentration zero): the
    // product-integration form must reproduce the
    // Epstein–Plesset transient uptake 8R²C∞√(πDt) — the
    // derivation that verifies the Eq. (14) prefactor
    // (prototype relative error 1.5e-15; assert < 1 %).
    let d = 2.42e-9;
    let r_fix = 40e-6;
    let c_inf = 0.2 * 0.822;
    let f = 1.0e6;
    let dt = 1.0 / (f * 200.0);
    let n_steps = 200 * 25;
    let mut hist = EllerFlynnHistory::new(d);
    for _ in 0..n_steps {
        hist.add_step(r_fix, r_fix, dt, -c_inf);
    }
    let t_tot = n_steps as f64 * dt;
    let m_model = -4.0 * (std::f64::consts::PI * d).sqrt() * hist.integral_j();
    let m_ep = 8.0 * r_fix * r_fix * c_inf * (std::f64::consts::PI * d * t_tot).sqrt();
    let rel = (m_model - m_ep).abs() / m_ep;
    assert!(
        rel < 0.01,
        "EF vs EP: model {m_model:e}, EP {m_ep:e}, rel {rel:e}"
    );
    // No-renewal limit: cumulative ratio vs the flat
    // per-cycle law is exactly √(2/N) at N = 25.
    let q0 = 8.0 * c_inf * (std::f64::consts::PI * d * 0.5 / f).sqrt() * r_fix * r_fix;
    let rn = m_model / (25.0 * q0);
    assert!(
        (rn - (2.0f64 / 25.0).sqrt()).abs() < 0.01,
        "r(25) = {rn}, bound {}",
        (2.0f64 / 25.0).sqrt()
    );
}

#[test]
fn test_vapour_rate_signs_and_cap() {
    // Undersaturated wall (p_sat > f·p_i): evaporation > 0.
    let evap = vapour_rate_hertz_knudsen(10e-6, -1.0, 310.0, 310.0, 0.0, 101_325.0, 0.0);
    assert!(evap > 0.0, "evaporation rate {evap}");
    // Above T_c: phase change off.
    let off = vapour_rate_hertz_knudsen(10e-6, -1.0, WATER_T_CRIT_K, 310.0, 0.0, 101_325.0, 0.0);
    assert_eq!(off, 0.0);
}

#[test]
fn test_integrator_quiescent_history() {
    // A bubble held at its equilibrium radius with C* = 1
    // and vapour off: F ≈ C_{g,w} − C∞ with C_{g,w} ≈ C0 at
    // p_i ≈ p0 + 2σ/R — slightly supersaturated wall, so the
    // signed flux must be a small OUTFLOW (dissolution
    // direction), and nothing may blow up over 5 cycles.
    let params = KreiderGainParams {
        vapour: false,
        saturation_ratio: 1.0,
        ..KreiderGainParams::default()
    };
    // Equilibrium content at R_eq = 5 µm (implicit-map-free
    // estimate: p_g = p0 + 2σ/R, ideal gas at 310 K).
    let r_eq = 5e-6;
    let p_g = params.ambient_pressure + 2.0 * params.surface_tension / r_eq;
    let volume = 4.0 / 3.0 * std::f64::consts::PI * r_eq.powi(3);
    let n0 = p_g * volume / (R_GAS * params.temperature);
    let mut model = KreiderGainModel::new(n0, params);
    let ppc = 200;
    let f = 1.0e6;
    let dt = 1.0 / (f * ppc as f64);
    let radius = vec![r_eq; 5 * ppc + 1];
    let radius_dot = vec![0.0; 5 * ppc + 1];
    let time: Vec<f64> = (0..=5 * ppc).map(|i| i as f64 * dt).collect();
    let diags = model.integrate_cycles(&radius, &radius_dot, &time, ppc, r_eq);
    assert_eq!(diags.len(), 5);
    let dn = model.gas_content_mol() - n0;
    assert!(
        dn < 0.0 && dn.abs() / n0 < 0.05,
        "quiescent dn/n0 = {}",
        dn / n0
    );
    assert_eq!(model.choke_events(), 0);
}
