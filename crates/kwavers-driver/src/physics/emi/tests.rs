//! Phase 3d emi slice — 8 lifted tests from the flat `src/emi.rs` (verbatim move).
//!
//! Eight pinning tests:
//! * `closer_cap_yields_smaller_loop_and_inductance` — geometric correctness of scene walker.
//! * `inductance_is_few_nh_for_mm_scale_loop` — locks `loop_inductance_nh(1.0) ≈ 1.26 nH = μ₀`.
//! * `trace_inductance_is_about_ten_nh_per_cm` — verifies the Grover/IPC rule of thumb
//!   for the partial-inductance model.
//! * `gate_and_recovery_losses_are_qvf` — locks `Q_g·V·f` and `Q_rr·V·f` scaling
//!   (linearity in `f`).
//! * `switching_loss_is_cv2f` — locks `f·C·V²` (parabolic `V` dependence).
//! * `bvd_load_drive_current_matches_device_rating` — locks the 1.5 A HV7355 peak + 3 V
//!   L·dI/dt coupling for the 10 nH / 5 ns envelope.
//! * `radiated_emi_scales_with_area_and_frequency` — locks the 4×area/+12 dB,
//!   2×frequency/+12 dB, 2×distance/−6 dB scaling laws (parses `radiated::radiated_emi_dbuv_m`).
//! * `cispr22_class_b_limit_context` — exercises the compliance oracle at the 30 MHz class B
//!   radiated limit.

use super::*;
use crate::board::{LayerId, NetId};
use crate::geom::{Nm, Point};
use crate::place::component::{Component, Placement};
use crate::place::footprint::{FootprintDef, PadDef, Role};
use crate::place::rotation::Rot;

fn ic_fp() -> FootprintDef {
    FootprintDef::new(
        "IC",
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ],
    )
}

fn cap_fp() -> FootprintDef {
    FootprintDef::new(
        "C",
        (Nm::from_mm(2.0), Nm::from_mm(1.0)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ],
    )
}

fn scene(cap_y: f64) -> (Vec<Component>, Vec<FootprintDef>) {
    let lib = vec![ic_fp(), cap_fp()];
    let vpp = NetId(0);
    let gnd = NetId(1);
    let ic = Component {
        fp: 0,
        nets: vec![Some(vpp), Some(gnd)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let cap = Component {
        fp: 1,
        nets: vec![Some(vpp), Some(gnd)],
        refdes: "C1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(0.0), Nm::from_mm(cap_y)),
            rot: Rot::R0,
        },
        assoc_ic: Some(0),
        locked: false,
        ..Default::default()
    };
    (vec![ic, cap], lib)
}

#[test]
fn closer_cap_yields_smaller_loop_and_inductance() {
    let (near, lib) = scene(2.0);
    let (far, _) = scene(10.0);
    let ln = commutation_loops(&near, &lib);
    let lf = commutation_loops(&far, &lib);
    assert_eq!(ln.len(), 1);
    assert_eq!(lf.len(), 1);
    assert!(
        ln[0].area_mm2 < lf[0].area_mm2,
        "a closer decoupling cap must enclose a smaller commutation loop ({:.1} vs {:.1} mm²)",
        ln[0].area_mm2,
        lf[0].area_mm2
    );
    assert!(ln[0].inductance_nh < lf[0].inductance_nh);
    assert!(ln[0].inductance_nh > 0.0);
}

#[test]
fn inductance_is_few_nh_for_mm_scale_loop() {
    // A 1 mm² loop ⇒ L ≈ μ₀·1mm ≈ 1.26 nH.
    let l = loop_inductance_nh(1.0);
    assert!((l - 1.256).abs() < 0.01, "expected ~1.26 nH, got {l:.3}");
}

#[test]
fn trace_inductance_is_about_ten_nh_per_cm() {
    // 10 mm of 0.25 mm trace ⇒ ~10 nH (rule of thumb 6–10 nH/cm).
    let l = trace_partial_inductance_nh(10.0e-3, 0.25e-3, 35e-6);
    assert!((6.0..=12.0).contains(&l), "expected ~10 nH/cm, got {l:.1}");
}

#[test]
fn gate_and_recovery_losses_are_qvf() {
    // 20 nC gate charge, 5 V, 2 MHz ⇒ 0.2 W gate drive.
    assert!((gate_drive_power_w(20e-9, 5.0, 2.0e6) - 0.2).abs() < 1e-3);
    // Both scale linearly with frequency.
    assert!(
        reverse_recovery_loss_w(5e-9, 150.0, 2.0e6)
            > reverse_recovery_loss_w(5e-9, 150.0, 1.0e6)
    );
}

#[test]
fn switching_loss_is_cv2f() {
    // 2 MHz, 25 pF output cap, 150 V ⇒ P = f·C·V² = 2e6·25e-12·150² ≈ 1.13 W per node.
    let p = switching_loss_w(2.0e6, 25e-12, 150.0);
    assert!((p - 1.125).abs() < 0.01, "expected ~1.13 W, got {p:.3}");
}

#[test]
fn bvd_load_drive_current_matches_device_rating() {
    // 50 pF charged 0→150 V in 5 ns ⇒ I = C·dV/dt = 1.5 A — the HV7355's peak rating.
    let i = capacitive_drive_current_a(50e-12, 150.0, 5e-9);
    assert!((i - 1.5).abs() < 0.01, "expected 1.5 A, got {i:.3}");
    // That 1.5 A through 10 nH at 5 ns rings ~3 V — small, but scales with loop inductance.
    let v = inductive_overshoot_v(10.0, 1.5, 5e-9);
    assert!((v - 3.0).abs() < 0.1, "expected ~3 V overshoot, got {v:.2}");
}

#[test]
fn radiated_emi_scales_with_area_and_frequency() {
    // HV7355 typical: 2 MHz, 1 mm² loop, 1.5 A peak, measured at 3 m.
    // E = 1.316e-14 · (2e6)² · 1e-6 · 1.5 / 3 ≈ 2.63e-8 V/m = 2.63e-2 µV/m ≈ −31.6 dBµV/m.
    let e = radiated_emi_dbuv_m(2.0e6, 1.0, 1.5, 3.0);
    assert!(
        (e - (-31.6)).abs() < 1.0,
        "expected ~−31.6 dBµV/m, got {e:.1}"
    );
    // Quadrupling loop area ⇒ +12 dB (4×).
    let e4 = radiated_emi_dbuv_m(2.0e6, 4.0, 1.5, 3.0);
    assert!(
        (e4 - e - 12.0).abs() < 0.1,
        "4× area must give +12 dB: expected {:.1}, got {e4:.1}",
        e + 12.0
    );
    // Doubling frequency ⇒ +6 dB (f² ⇒ 4×).
    let e2f = radiated_emi_dbuv_m(4.0e6, 1.0, 1.5, 3.0);
    assert!(
        (e2f - e - 12.0).abs() < 0.1,
        "2× frequency must give +12 dB: expected {:.1}, got {e2f:.1}",
        e + 12.0
    );
    // Doubling distance ⇒ −6 dB (1/r).
    let e6m = radiated_emi_dbuv_m(2.0e6, 1.0, 1.5, 6.0);
    assert!(
        (e - e6m - 6.0).abs() < 0.1,
        "2× distance must give −6 dB: got {:.1} vs {e6m:.1}",
        e
    );
    // Degenerate inputs.
    assert!(radiated_emi_dbuv_m(0.0, 1.0, 1.5, 3.0).is_infinite());
    assert!(radiated_emi_dbuv_m(2.0e6, 0.0, 1.5, 3.0).is_infinite());
}

#[test]
fn cispr22_class_b_limit_context() {
    // CISPR 22 Class B radiated emission limit at 30 MHz, 3 m: 30 dBµV/m.
    // Threshold loop area at 30 MHz, 1 A: A = 3.16e-5 / (1.316e-14·(30e6)²/3) ≈ 8 mm².
    // A 10 mm² loop at 30 MHz, 1 A: E ≈ 1.316e-14·(30e6)²·1e-5·1/3 ≈ 31.9 dBµV/m — above
    // the limit, showing why minimizing loop area is critical.
    let e30 = radiated_emi_dbuv_m(30.0e6, 10.0, 1.0, 3.0);
    assert!(
        e30 > 30.0,
        "10 mm² loop at 30 MHz must exceed CISPR 22 class B limit"
    );
    // A 1 mm² loop at 30 MHz, 1 A: E ≈ 11.9 dBµV/m — well within the limit.
    let e30_small = radiated_emi_dbuv_m(30.0e6, 1.0, 1.0, 3.0);
    assert!(
        e30_small < 30.0,
        "1 mm² loop at 30 MHz should be within the limit"
    );
}
