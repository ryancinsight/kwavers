//! Consolidated tests for the `driver` slice (Phase 4g carve-out): the pulser loss model, the
//! reactive/matching math, the thermal-duty + power-rating limits, the frequency sweep and the
//! cross-IC comparison. Moved verbatim from the flat `src/driver.rs` `mod tests` block; `super::*`
//! resolves the slice facade.

use super::*;

fn op() -> PulserOp {
    PulserOp {
        drive_hz: 2.0e6,
        duty: 1.0,
        c_load_f: 50e-12,
        v_pp: 150.0,
        r_on_ohm: 8.0,      // HV7355 P-channel on-resistance
        r_series_ohm: 56.0, // series damping resistor
        q_g_c: 20e-9,
        v_gate: 5.0,
        q_rr_c: 5e-9,
    }
}

#[test]
fn dynamic_loss_is_cv2f_and_splits_by_resistance() {
    let d = pulser_dissipation(&op());
    // P_dyn = 1·2e6·50e-12·150² = 2.25 W (CW).
    assert!(
        (d.dynamic_total - 2.25).abs() < 1e-3,
        "got {}",
        d.dynamic_total
    );
    // Device gets R_on/(R_on+R_series) = 8/64 = 12.5% ⇒ ~0.281 W; the 56 Ω resistor gets the rest.
    assert!((d.dynamic_device - 2.25 * 8.0 / 64.0).abs() < 1e-3);
    assert!((d.dynamic_series_r - 2.25 * 56.0 / 64.0).abs() < 1e-3);
    // Most of the dynamic loss is in the damping resistor, not the device — a real thermal insight.
    assert!(d.dynamic_series_r > d.dynamic_device * 5.0);
}

#[test]
fn duty_cycle_scales_dissipation_linearly() {
    let mut o = op();
    o.duty = 0.01; // 1% burst duty (imaging PRF)
    let d = pulser_dissipation(&o);
    assert!((d.dynamic_total - 0.0225).abs() < 1e-4, "1% duty ⇒ 22.5 mW");
    // Gate + recovery also scale with duty.
    assert!((d.gate - 0.01 * 20e-9 * 5.0 * 2.0e6).abs() < 1e-9);
}

#[test]
fn device_total_sums_its_three_components() {
    let d = pulser_dissipation(&op());
    assert!((d.device_total - (d.dynamic_device + d.gate + d.recovery)).abs() < 1e-9);
}

#[test]
fn tuning_inductor_resonates_out_c0() {
    // 50 pF at 2 MHz ⇒ L = 1/((2π·2e6)²·50e-12) ≈ 127 µH.
    let l = tuning_inductor_h(50e-12, 2.0e6);
    assert!((l - 126.65e-6).abs() < 1e-6, "got {} µH", l * 1e6);
    // Resonance check: 1/(2π√(LC)) ≈ f0.
    let f = 1.0 / (2.0 * std::f64::consts::PI * (l * 50e-12).sqrt());
    assert!((f - 2.0e6).abs() < 1e3);
}

#[test]
fn damping_q_and_ringdown() {
    // 56 Ω with 50 pF at 2 MHz: Xc = 1/(2π·2e6·50e-12) ≈ 1592 Ω ⇒ Q ≈ 28 — very lightly damped!
    let q = load_quality_factor(50e-12, 2.0e6, 56.0);
    assert!((q - 28.4).abs() < 0.5, "got Q={q:.1}");
    // For Q≈1 (short pulse) the damping resistor must be ~1.6 kΩ, far above 56 Ω.
    let r = damping_resistor_ohm(50e-12, 2.0e6, 1.0);
    assert!((1500.0..1700.0).contains(&r), "got R={r:.0}");
    // Ringdown ≈ Q/π cycles.
    assert!((ringdown_cycles(q) - q / std::f64::consts::PI).abs() < 1e-9);
}

#[test]
fn max_safe_duty_scales_with_thermal_headroom() {
    // At 10 % duty the board rises 20 K; the 40 K limit allows 2× ⇒ 20 % max duty.
    assert!((max_safe_duty(0.10, 20.0, 40.0) - 0.20).abs() < 1e-9);
    // Headroom beyond CW saturates at 1.0.
    assert_eq!(max_safe_duty(0.10, 1.0, 40.0), 1.0);
}

#[test]
fn chip_rating_and_overload_detection() {
    use crate::geom::{Nm, Point};
    use crate::place::component::{Component, Placement};
    use crate::place::footprint::{FootprintDef, Role};
    use crate::place::rotation::Rot;
    assert_eq!(chip_power_rating_w("R0402"), Some(0.063));
    assert_eq!(chip_power_rating_w("R_1206_3216Metric"), Some(0.25));
    assert_eq!(chip_power_rating_w("HV7355K6-G"), None);
    // A part dissipating 0.2 W: an 0402 (63 mW) is overloaded}; a 1206 (250 mW) is fine.
    let comp = |fp| Component {
        fp,
        nets: vec![],
        refdes: "R1".into(),
        placement: Placement {
            pos: Point::new(Nm(0), Nm(0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let lib = vec![
        FootprintDef::new(
            "R0402",
            (Nm::from_mm(1.0), Nm::from_mm(0.5)),
            Role::Passive,
            vec![],
        ),
        FootprintDef::new(
            "R1206",
            (Nm::from_mm(3.2), Nm::from_mm(1.6)),
            Role::Passive,
            vec![],
        ),
    ];
    let r0402 = power_rating_check(&[comp(0)], &lib, |_| 0.2);
    assert_eq!(r0402.overloaded.len(), 1, "0402 overloaded at 0.2 W");
    assert!(!r0402.pass);
    let r1206 = power_rating_check(&[comp(1)], &lib, |_| 0.2);
    assert!(r1206.pass, "1206 holds 0.2 W");
}

#[test]
fn matching_lifts_efficiency() {
    // Direct drive: P_loss ≈ reactive CV²f dominates the small acoustic power ⇒ low η.
    let p_ac = 0.05; // 50 mW radiated (example)
    let p_reactive = reactive_drive_power_w(50e-12, 150.0, 2.0e6, 1.0); // 2.25 W
    let eta_direct = driver_efficiency(p_ac, p_reactive);
    assert!(
        eta_direct < 0.05,
        "direct capacitive drive is inefficient: {eta_direct:.3}"
    );
    // Matched: the tuning inductor recovers the reactive energy, leaving ~10% residual loss ⇒
    // efficiency rises sharply.
    let eta_matched = driver_efficiency(p_ac, 0.1 * p_reactive);
    assert!(
        eta_matched > eta_direct * 3.0,
        "matching network lifts efficiency"
    );
}

#[test]
fn sweep_produces_n_points() {
    let sweep = sweep_driver_loss(
        1.0e6, 3.0e6, 5, 50e-12, 150.0, 8.0, 56.0, 20e-9, 5.0, 5e-9, 0.05, 40.0, 1.0,
    );
    assert_eq!(sweep.len(), 5);
    // Points should be in ascending frequency order.
    for w in sweep.windows(2) {
        assert!(w[0].freq_hz < w[1].freq_hz);
    }
    // All efficiencies should be >0 and <1.
    for p in &sweep {
        assert!(p.efficiency > 0.0 && p.efficiency < 1.0);
    }
}

#[test]
fn find_best_freq_returns_min_loss() {
    let sweep = sweep_driver_loss(
        1.0e6, 3.0e6, 5, 50e-12, 150.0, 8.0, 56.0, 20e-9, 5.0, 5e-9, 0.05, 40.0, 1.0,
    );
    let best = find_best_freq(&sweep).expect("sweep has points");
    // All others should have >= total_device_w.
    for p in &sweep {
        assert!(p.total_device_w >= best.total_device_w - 1e-12);
    }
}

#[test]
fn compare_driver_ics_ranks_by_efficiency() {
    // HV7355 (R_on=8, Q_g=20nC) vs STHV748S (R_on=3.5, Q_g=30nC)
    let comps = compare_driver_ics_at(
        2.0e6,
        50e-12,
        150.0,
        56.0,
        5.0,
        5e-9,
        0.05,
        40.0,
        1.0,
        &[("HV7355", 8.0, 20e-9), ("STHV748S", 3.5, 30e-9)],
    );
    assert_eq!(comps.len(), 2);
    // At this operating point STHV748S's larger Q_g (30 nC) adds 0.1 W
    // gate loss vs HV7355 (20 nC), offsetting the R_on advantage near
    // 1% carrier-frame efficiency. Both results are in the same ballpark.
    let effs: Vec<_> = comps.iter().map(|c| c.efficiency).collect();
    let min_eff = effs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_eff = effs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_eff - min_eff < 0.01,
        "efficiency spread should be tight: {:.4}..{:.4}",
        min_eff,
        max_eff
    );
    assert!(min_eff > 0.0, "all efficiencies positive");
}

#[test]
fn switching_node_ringing_matches_characteristic_impedance() {
    // HV7355 typical: 10 nH loop, 25 pF node cap, 1.5 A peak.
    // V_ring = I · sqrt(L/C) = 1.5 · sqrt(10e-9/25e-12) = 1.5 · 20 = 30 V.
    let v = switching_node_ringing_v(1.5, 10.0, 25e-12);
    assert!((v - 30.0).abs() < 0.5, "expected ~30 V ring, got {v:.2}");
    // More loop inductance ⇒ higher ring (proportional to sqrt(L)).
    let v2 = switching_node_ringing_v(1.5, 40.0, 25e-12);
    assert!(
        (v2 - 2.0 * v).abs() < 0.5,
        "4× inductance must double the ringing voltage"
    );
    // Degenerate: zero inductance or capacitance ⇒ no ringing.
    assert_eq!(switching_node_ringing_v(1.5, 0.0, 25e-12), 0.0);
    assert_eq!(switching_node_ringing_v(1.5, 10.0, 0.0), 0.0);
}

#[test]
fn thermally_derated_efficiency_decreases_with_temperature() {
    let op = op(); // from the existing helper: 2 MHz, 50 pF, 150 V, etc.
    let eta_cold = thermally_derated_efficiency(&op, 298.0, 6.0e-3, 0.05);
    // At T_j = 378 K (80 K above reference): R_on × (1 + 0.006×80) = 1.48×
    // ⇒ more device dissipation ⇒ lower efficiency.
    let eta_hot = thermally_derated_efficiency(&op, 378.0, 6.0e-3, 0.05);
    assert!(
        eta_cold >= eta_hot,
        "hot Rds_on must lower or equal efficiency: cold={eta_cold:.4}, hot={eta_hot:.4}"
    );
    // At reference temperature, derated efficiency equals the cold model.
    let eta_ref = thermally_derated_efficiency(&op, 298.0, 6.0e-3, 0.05);
    let eta_baseline = driver_efficiency(0.05, pulser_dissipation(&op).device_total);
    assert!(
        (eta_ref - eta_baseline).abs() < 1e-9,
        "at T_ref derated efficiency must match cold model"
    );
}
