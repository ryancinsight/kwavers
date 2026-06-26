//! Consolidated tests for the `optim` slice (Phase 4h carve-out): the full design-point evaluation
//! and the standalone thermal/breakdown/track-resistance kernels. Moved verbatim from the flat
//! `src/optim.rs` `mod tests` block; `super::*` resolves the slice facade.

use super::*;
use crate::driver::PulserOp;

fn hv7355_op() -> PulserOp {
    PulserOp {
        drive_hz: 2.0e6,
        duty: 0.5,
        c_load_f: 50e-12,
        v_pp: 150.0,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        q_g_c: 20e-9,
        v_gate: 5.0,
        q_rr_c: 5e-9,
    }
}

fn standard_array() -> ArrayGeometry {
    ArrayGeometry::new(16, 4.3e-3 / 15.0, 10.0e-3, 30.0, 1540.0, 2.0e6)
}

fn default_pdn() -> PdnConfig {
    PdnConfig {
        caps: vec![
            (10e-6, 20e-3, 10e-9),   // bulk: 10 µF, 20 mΩ ESR, 10 nH ESL
            (100e-9, 50e-3, 0.5e-9), // local: 100 nF, 50 mΩ ESR, 0.5 nH ESL
        ],
        v_ripple: 0.5,
        i_transient_a: 1.5,
        loop_area_mm2: 2.0,
    }
}

#[test]
fn design_report_is_fully_populated() {
    let op = hv7355_op();
    let array = standard_array();
    let thermal = ThermalContext::default();
    let pdn = default_pdn();
    let emi = EmiContext::default();

    let r = evaluate_design_point(&op, &array, &thermal, &pdn, &emi, 0.05, 6.0, 1.48e6);

    assert!(r.p_device_cold_w > 0.0, "cold dissipation must be positive");
    assert!(
        r.efficiency_cold > 0.0 && r.efficiency_cold < 1.0,
        "cold efficiency must be in (0,1)"
    );
    assert!(r.switching_ring_v > 0.0, "ringing voltage must be positive");
    assert!(r.t_j_k > 298.0, "junction temperature must exceed ambient");
    assert!(
        r.focal_pressure_gain > 1.0,
        "16-element array must amplify pressure"
    );
    assert!(
        (r.array_factor_at_steer - 1.0).abs() < 1e-6,
        "array factor at steer angle must be ~1"
    );
    assert!(r.pdn_z_target_ohm > 0.0, "PDN target impedance must be set");
    assert!(
        r.anti_resonance_hz.is_finite(),
        "anti-resonance must be a finite frequency"
    );
}

#[test]
fn thermal_derating_lowers_efficiency() {
    let op = hv7355_op();
    let array = standard_array();
    let cold = ThermalContext::default();
    let hot = ThermalContext {
        board_rise_k: 50.0, // severe board heat
        ..cold
    };
    let r_cold = evaluate_design_point(
        &op,
        &array,
        &cold,
        &default_pdn(),
        &EmiContext::default(),
        0.05,
        6.0,
        1.48e6,
    );
    let r_hot = evaluate_design_point(
        &op,
        &array,
        &hot,
        &default_pdn(),
        &EmiContext::default(),
        0.05,
        6.0,
        1.48e6,
    );
    assert!(
        r_hot.t_j_k > r_cold.t_j_k,
        "hot board must raise T_j: {:.1} vs {:.1}",
        r_hot.t_j_k,
        r_cold.t_j_k
    );
    assert!(
        r_hot.efficiency_derated <= r_cold.efficiency_derated,
        "hot board must lower or equal derated efficiency"
    );
}

#[test]
fn max_safe_duty_is_below_one_at_nominal() {
    let op = hv7355_op();
    let thermal = ThermalContext {
        t_ambient_k: 298.0,
        board_rise_k: 0.0,
        theta_jc_k_per_w: 40.0,
        alpha_rds_per_k: 6.0e-3,
        t_j_max_k: 423.0,
    };
    let d_max = max_safe_duty_thermal(&op, &thermal);
    assert!(
        d_max > 0.0 && d_max <= 1.0,
        "max safe duty must be in (0, 1], got {d_max:.3}"
    );
}

#[test]
fn ringing_breakdown_check_detects_margin_violation() {
    // 150 V supply, 30 V ring, 200 V breakdown, 10% margin → limit = 180 V.
    // 150 + 30 = 180 V = limit → exactly at edge (not exceeded).
    assert!(
        !ringing_exceeds_breakdown(150.0, 30.0, 200.0, 0.10),
        "180 V peak should not exceed 180 V limit"
    );
    // 31 V ring → 181 V > 180 V limit → violation.
    assert!(
        ringing_exceeds_breakdown(150.0, 31.0, 200.0, 0.10),
        "181 V peak must exceed 180 V limit"
    );
}

#[test]
fn hot_track_resistance_exceeds_cold() {
    // 100 mm of 0.25 mm, 1 oz track: R_dc ≈ 0.193 Ω at 20 °C (293 K).
    // At 100 °C (373 K): ΔT = 80 K, α = 3.93e-3 → R_hot = 0.193 × 1.314 ≈ 0.254 Ω.
    let r_cold = hot_track_resistance(0.1, 0.25e-3, 1.0, 293.0);
    let r_hot = hot_track_resistance(0.1, 0.25e-3, 1.0, 373.0);
    assert!(
        r_hot > r_cold,
        "hot track resistance must exceed cold: {r_hot:.4} > {r_cold:.4}"
    );
    assert!(
        (r_hot / r_cold - 1.314).abs() < 0.02,
        "expected ~31.4% increase, got {:.1}%",
        (r_hot / r_cold - 1.0) * 100.0
    );
}
