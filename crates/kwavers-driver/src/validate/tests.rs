//! Tests for the `validate` slice (Phase 4j).
//!
//! NOTE — RECONSTRUCTED SUITE: the original 662-line `mod tests` block was lost when the flat
//! `src/validate.rs` file was removed (a tool auto-resolving the transient `validate.rs`↔`validate/`
//! module ambiguity) before its test body could be extracted into this slice. This suite is a
//! from-contract reconstruction: every assertion is derived analytically from each function's
//! documented behaviour, covering the `Check`/`PhysicsReport` primitives, the board-geometry checks,
//! and the kwavers-beam driver→transducer seam. It should be cross-checked against the original
//! intent if a copy resurfaces.

use super::*;
use crate::board::{Board, LayerId, NetClassKind, Track, Via, ViaKind};
use crate::geom::{GridSpec, Nm, Point};
use crate::manifest::{
    DriverManifest, EnergyBudgetInputs, EnergyBudgetReport, ResistorPackage, StimulationProgram,
    TileStimulationProfile,
};
use crate::rules::DesignRules;

fn spec() -> GridSpec {
    GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap()
}

fn track(net: crate::board::NetId, x0: f64, y0: f64, x1: f64, y1: f64, w_mm: f64) -> Track {
    Track {
        start: Point::new(Nm::from_mm(x0), Nm::from_mm(y0)),
        end: Point::new(Nm::from_mm(x1), Nm::from_mm(y1)),
        width: Nm::from_mm(w_mm),
        layer: LayerId(0),
        net,
    }
}

fn via(net: crate::board::NetId, x: f64, y: f64, kind: ViaKind, filled: bool) -> Via {
    Via {
        pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.6),
        net,
        from: LayerId(0),
        to: LayerId(1),
        kind,
        filled,
    }
}

// ── Check / PhysicsReport primitives ───────────────────────────────────────────────────────────

#[test]
fn check_directions_and_margins() {
    let u = Check::upper("rise", 22.0, 30.0, "K");
    assert!(u.pass && (u.margin - 8.0).abs() < 1e-9);
    let u2 = Check::upper("rise", 35.0, 30.0, "K");
    assert!(!u2.pass && (u2.margin + 5.0).abs() < 1e-9);
    let l = Check::lower("creepage", 0.6, 0.4, "mm");
    assert!(l.pass && (l.margin - 0.2).abs() < 1e-9);
    let l2 = Check::lower("creepage", 0.3, 0.4, "mm");
    assert!(!l2.pass && (l2.margin + 0.1).abs() < 1e-9);
}

#[test]
fn physics_report_all_pass_iff_every_check_passes() {
    let ok = PhysicsReport::new(vec![
        Check::upper("a", 1.0, 2.0, ""),
        Check::lower("b", 2.0, 1.0, ""),
    ]);
    assert!(ok.all_pass);
    let bad = PhysicsReport::new(vec![
        Check::upper("a", 1.0, 2.0, ""),
        Check::upper("c", 3.0, 2.0, ""),
    ]);
    assert!(!bad.all_pass);
    assert!(PhysicsReport::new(vec![]).all_pass, "vacuous report passes");
}

#[test]
fn core_checks_assembles_five_directional_checks() {
    let pass = core_checks(20.0, 30.0, 0.5, 0.1, 0.3, 0, 0);
    assert_eq!(pass.len(), 5);
    assert!(PhysicsReport::new(pass).all_pass);
    // ampacity headroom is a lower-bound; negative ⇒ fail. via-adjacency/dangling are upper(.,0).
    let fail = core_checks(20.0, 30.0, -0.2, 0.1, 0.3, 1, 2);
    let report = PhysicsReport::new(fail);
    assert!(!report.all_pass);
    assert!(!report.checks[1].pass, "negative ampacity margin fails");
    assert!(
        !report.checks[3].pass,
        "via-adjacency 1 fails upper bound 0"
    );
}

// ── Board-geometry checks ──────────────────────────────────────────────────────────────────────

#[test]
fn via_census_tallies_kinds_and_vippo() {
    let mut b = Board::new(spec());
    let n = b.add_net("N", NetClassKind::Signal);
    b.vias.push(via(n, 1.0, 1.0, ViaKind::Through, false));
    b.vias.push(via(n, 2.0, 1.0, ViaKind::Blind, true));
    b.vias.push(via(n, 3.0, 1.0, ViaKind::Micro, true));
    let c = via_census(&b);
    assert_eq!(c.through, 1);
    assert_eq!(c.blind, 1);
    assert_eq!(c.micro, 1);
    assert_eq!(c.buried, 0);
    assert_eq!(c.vippo, 2, "two filled vias are VIPPO");
}

#[test]
fn microvia_aspect_check_is_vacuous_without_micro_and_grades_with_one() {
    let rules = DesignRules::holohv();
    let mut b = Board::new(spec());
    // No micro-vias ⇒ AR = 0 ≤ limit, passes.
    assert!(microvia_aspect_check(&b, &rules, 1.0).pass);

    let n = b.add_net("N", NetClassKind::Signal);
    b.vias.push(via(n, 1.0, 1.0, ViaKind::Micro, true));
    let drill_mm = rules.microvia_drill.to_mm();
    // build_up just under limit·drill ⇒ AR ≤ limit ⇒ pass; well over ⇒ fail.
    let under = (rules.max_microvia_ar - 0.1) * drill_mm;
    let over = (rules.max_microvia_ar + 0.5) * drill_mm;
    assert!(microvia_aspect_check(&b, &rules, under).pass);
    assert!(!microvia_aspect_check(&b, &rules, over).pass);
}

#[test]
fn net_length_and_group_skew() {
    let mut b = Board::new(spec());
    let a = b.add_net("A", NetClassKind::Signal);
    let c = b.add_net("C", NetClassKind::Signal);
    b.tracks.push(track(a, 1.0, 1.0, 6.0, 1.0, 0.2)); // 5 mm
    b.tracks.push(track(c, 1.0, 3.0, 4.0, 3.0, 0.2)); // 3 mm
    assert!((net_length_mm(&b, a) - 5.0).abs() < 1e-6);
    assert!((net_length_mm(&b, c) - 3.0).abs() < 1e-6);
    assert!((group_skew_mm(&b, &[a, c]) - 2.0).abs() < 1e-6);
    // A single routed net has zero skew.
    assert_eq!(group_skew_mm(&b, &[a]), 0.0);
}

#[test]
fn min_hv_spacing_finds_closest_different_net_hv_pair() {
    let mut b = Board::new(spec());
    let hv = b.add_net("VPP", NetClassKind::Hv);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    // Vias (not pads) so the <1 mm same-component pad exemption does not apply; 3 mm apart.
    b.vias.push(via(hv, 2.0, 2.0, ViaKind::Through, false));
    b.vias.push(via(gnd, 5.0, 2.0, ViaKind::Through, false));
    assert!((min_hv_spacing_mm(&b) - 3.0).abs() < 1e-6);
    // No HV pair ⇒ infinity.
    let mut b2 = Board::new(spec());
    let s1 = b2.add_net("S1", NetClassKind::Signal);
    let s2 = b2.add_net("S2", NetClassKind::Signal);
    b2.vias.push(via(s1, 2.0, 2.0, ViaKind::Through, false));
    b2.vias.push(via(s2, 5.0, 2.0, ViaKind::Through, false));
    assert!(min_hv_spacing_mm(&b2).is_infinite());
}

#[test]
fn worst_ampacity_margin_flags_undersized_track() {
    let mut b = Board::new(spec());
    let pwr = b.add_net("PWR", NetClassKind::Power);
    b.tracks.push(track(pwr, 1.0, 1.0, 6.0, 1.0, 0.1)); // narrow 0.1 mm
                                                        // 2 A at 10 °C rise on 1 oz needs far more than 0.1 mm ⇒ negative margin.
    let margin = worst_ampacity_margin_mm(&b, |n| if n == pwr { 2.0 } else { 0.0 }, 10.0, 1.0);
    assert!(
        margin < 0.0,
        "undersized track has negative headroom: {margin}"
    );
    // Zero current everywhere ⇒ no carrying net binds ⇒ infinity.
    let none = worst_ampacity_margin_mm(&b, |_| 0.0, 10.0, 1.0);
    assert!(none.is_infinite());
}

// ── Kwavers-beam driver→transducer seam ─────────────────────────────────────────────────────────

fn four_tile_v2_manifest() -> DriverManifest {
    let mut tile_profiles = Vec::new();
    for i in 0..4 {
        tile_profiles.push(TileStimulationProfile::from_article_with(
            1.0e3 + i as f64 * 50.0,
            i as f64 * 250.0e-3,
            i as f64 * 90.0,
            25.0e-6 + i as f64 * 5.0e-6,
        ));
    }
    DriverManifest {
        hv_board: "hv7355_driver_stack.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..96).map(|i| format!("TX_{i}")).collect(),
        programming: "fpga:JTAG=TCK,TMS,TDI,TDO; stack-bus=4×24-lane".into(),
        aperture_m: 4.3e-3 * 95.0 / 15.0,
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        stimulation: None,
        tile_profiles,
    }
}

fn v2_budget(m: &DriverManifest) -> EnergyBudgetReport {
    let inputs = EnergyBudgetInputs {
        c_load_f: 50e-12,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        ampacity_headroom_a: 8.0,
        damping_footprint: ResistorPackage::Smd4527,
    };
    m.validate_v2_energy_budget(inputs)
        .expect("well-sized v2 board")
}

/// A single-tile (24-lane) manifest carrying a legacy stim block — NOT full-stack v2.
fn non_v2_manifest() -> DriverManifest {
    DriverManifest {
        hv_board: "hv7355_tile.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..24).map(|i| format!("TX_{i}")).collect(),
        programming: "fpga:JTAG".into(),
        aperture_m: 4.3e-3,
        frequency_hz: 2.0e6,
        sound_speed_m_s: 1540.0,
        focal_m: 10.0e-3,
        timing_step_s: 5.0e-9,
        stimulation: Some(StimulationProgram::article_default()),
        tile_profiles: Vec::new(),
    }
}

#[test]
fn beam_step_built_from_full_stack_v2_has_derived_geometry() {
    let m = four_tile_v2_manifest();
    let budget = v2_budget(&m);
    let step = manifest_to_kwavers_beam_step(&m, &budget).expect("v2 pre-step");
    assert_eq!(step.lanes, 96);
    // wavelength = c/f; f_number = focal/aperture; pitch = aperture/(lanes-1).
    assert!((step.wavelength_m - 1540.0 / 2.0e6).abs() < 1e-12);
    assert!((step.f_number - m.focal_m / m.aperture_m).abs() < 1e-9);
    assert!((step.pitch_m - m.aperture_m / 95.0).abs() < 1e-9);
    assert_eq!(step.resistor_margin_w.len(), 4, "one margin per HV tile");
}

#[test]
fn beam_step_rejects_non_full_stack_v2() {
    let m = four_tile_v2_manifest();
    let budget = v2_budget(&m);
    let err = manifest_to_kwavers_beam_step(&non_v2_manifest(), &budget).unwrap_err();
    assert!(err.contains("full-stack v2"), "got: {err}");
}

#[test]
fn validate_against_budget_produces_full_report_for_v2() {
    let m = four_tile_v2_manifest();
    let budget = v2_budget(&m);
    let v = validate_against_budget(&m, &budget).expect("v2 validation");
    assert_eq!(v.step.lanes, 96);
    assert!(v.focal_pressure_pa > 0.0, "positive focal pressure");
    assert!(v.mechanical_index >= 0.0);
    assert!(v.isppa_w_cm2 > 0.0);
    assert!(v.axial_extent_mm > 0.0 && v.lateral_extent_mm > 0.0);
    // Article-class ~half-wavelength pitch ⇒ grating-lobe-free over ±90°.
    assert!(v.grating_lobe_free);
    // Four kwavers safety checks aggregated; margin vector mirrors the 4 tiles.
    assert_eq!(v.report.checks.len(), 4);
    assert_eq!(v.resistor_margin_w.len(), 4);
}

#[test]
fn validate_against_budget_rejects_non_v2_manifest() {
    let m = four_tile_v2_manifest();
    let budget = v2_budget(&m);
    assert!(validate_against_budget(&non_v2_manifest(), &budget).is_err());
}
