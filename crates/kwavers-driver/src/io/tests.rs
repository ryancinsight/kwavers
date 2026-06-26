//! Consolidated test surface for the [`crate::io`] slice.
//!
//! Phase 4a carve: the 5 unit tests in `src/io.rs::mod tests` migrated verbatim into this single
//! slice-wide test module. `pub use` re-exports at the slice root bring every public symbol into
//! `use super::*` scope identically to how the prior flat-module tests did.

use super::*;

use crate::board::{Board, LayerId, NetClassKind, Zone, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::component::{Component, Placement};
use crate::place::footprint::{FootprintDef, PadDef, Role};
use crate::place::rotation::Rot;
use crate::rules::{CreepageRule, DesignRules};

#[test]
fn emitted_pcb_has_well_formed_structure() {
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let n = board.add_net("VPP", NetClassKind::Hv);
    let gnd = board.add_net("GND", NetClassKind::Ground);
    let lib = vec![FootprintDef::new(
        "PART",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(1.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.6), Nm::from_mm(0.6)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    )];
    let comps = vec![Component {
        fp: 0,
        nets: vec![Some(n)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    board.tracks.push(crate::board::Track {
        start: Point::new(Nm::from_mm(11.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(15.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.3),
        layer: LayerId(0),
        net: n,
    });
    board.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(19.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(19.0), Nm::from_mm(19.0)),
            Point::new(Nm::from_mm(1.0), Nm::from_mm(19.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let pcb = write_kicad_pcb(&board, &comps, &lib, &DesignRules::holohv());

    // Balanced parentheses (well-formed S-expression).
    let opens = pcb.matches('(').count();
    let closes = pcb.matches(')').count();
    assert_eq!(opens, closes, "parentheses must balance");
    assert_eq!(
        duplicate_pcb_uuids(&pcb),
        Vec::<String>::new(),
        "emitted KiCad object UUIDs must be unique"
    );
    // Required stanzas present with values.
    assert!(pcb.starts_with("(kicad_pcb"));
    assert!(pcb.contains("(net 1 \"VPP\")"));
    assert!(pcb.contains("\"kicad-routing:PART\""));
    assert!(pcb.contains("(property \"Reference\" \"U1\""));
    assert!(pcb.contains("(segment"));
    assert!(pcb.contains("\"Edge.Cuts\""));
    assert!(
        pcb.contains("(connect_pads yes (clearance 0.130))"),
        "zone clearance must follow the design rules rather than overriding KiCad to zero"
    );
    // The HV track carries net 1 (VPP) at 0.3 mm.
    assert!(pcb.contains("(width 0.3000) (layer \"F.Cu\") (net 1)"));
}

#[test]
fn flagged_save_stamps_a_fab_layer_drc_fail_banner() {
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    let board = Board::new(spec);
    let path =
        std::env::temp_dir().join(format!("flagged_drc_{}.kicad_pcb", std::process::id()));
    save_kicad_pcb_flagged(
        &path,
        &board,
        &[],
        &[],
        &DesignRules::holohv(),
        "illegal routed capacity",
    )
    .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    // The board is unmistakably marked as a failing inspection artifact …
    let banner = text
        .lines()
        .find(|l| l.contains("DRC FAIL"))
        .expect("a DRC FAIL banner must be stamped");
    assert!(
        banner.contains("(gr_text"),
        "banner is a graphic text object"
    );
    // … on the F.Fab documentation layer, so it carries no silk-over-copper / silk-edge DRC.
    assert!(
        banner.contains("\"F.Fab\"") && !banner.contains("SilkS"),
        "banner must sit on F.Fab, not silkscreen: {banner}"
    );
    // Still a well-formed S-expression with unique UUIDs.
    assert_eq!(text.matches('(').count(), text.matches(')').count());
    assert_eq!(duplicate_pcb_uuids(&text), Vec::<String>::new());
}

#[test]
fn save_kicad_project_writes_dru_and_pro_alongside_pcb() {
    let spec =
        GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let board = Board::new(spec);
    let rules = DesignRules::holohv();
    let creepage = CreepageRule::holohv();
    let tmp = std::env::temp_dir().join("kicad_routing_project_test");
    std::fs::create_dir_all(&tmp).unwrap();
    let pcb = tmp.join("testboard.kicad_pcb");
    save_kicad_project(&pcb, &board, &[], &[], &rules, &creepage).unwrap();
    assert!(pcb.exists(), ".kicad_pcb must be written");
    assert!(
        tmp.join("testboard.kicad_dru").exists(),
        ".kicad_dru must be written"
    );
    assert!(
        tmp.join("testboard.kicad_pro").exists(),
        ".kicad_pro must be written"
    );
    let dru = std::fs::read_to_string(tmp.join("testboard.kicad_dru")).unwrap();
    assert!(dru.contains("0.130mm"), "DRU must encode holohv clearance");
}

#[test]
fn write_kicad_dru_encodes_engine_rules() {
    let rules = DesignRules::holohv();
    let creepage = CreepageRule::holohv();
    let text = write_kicad_dru(&rules, &creepage);
    assert!(text.contains("(version 1)"));
    assert!(
        text.contains("(rule \"track_min\""),
        "track_min rule must be present"
    );
    assert!(
        text.contains("(rule \"clearance_min\""),
        "clearance_min rule must be present"
    );
    assert!(
        text.contains("(rule \"HV_creepage\""),
        "HV creepage rule must be present"
    );
    assert!(text.contains("0.130mm"));
}

#[test]
fn write_kicad_pro_encodes_basename_and_rules() {
    let rules = DesignRules::holohv();
    let text = write_kicad_pro("tile", &rules);
    assert!(text.contains("\"tile.kicad_pro\""));
    assert!(text.contains("lib_footprint_issues"));
    assert!(text.contains("lib_footprint_mismatch"));
    assert!(text.contains("lib_symbol_issues"));
    // 0.130 mm clearance is the holohv setting.
    assert!(text.contains("0.130"));
}

#[test]
fn emitted_schematic_has_symbols_labels_and_balances() {
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let n = board.add_net("VPP", NetClassKind::Hv);
    let lib = vec![FootprintDef::new(
        "PART",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(1.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.6), Nm::from_mm(0.6)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    )];
    let comps = vec![Component {
        fp: 0,
        nets: vec![Some(n)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    let sch = write_kicad_sch(&board, &comps, &lib);
    assert_eq!(
        sch.matches('(').count(),
        sch.matches(')').count(),
        "parentheses must balance"
    );
    assert!(sch.starts_with("(kicad_sch"));
    assert!(sch.contains("(lib_symbols"));
    assert!(sch.contains("(symbol \"kicad-routing:PART\""));
    assert!(sch.contains("(property \"Reference\" \"U1\""));
    assert!(sch.contains("(global_label \"VPP\""));
    assert!(sch.contains("(sheet_instances"));
    // (Boundary-case clamp coverage is in the next test below; the
    // `emitted_schematic` test above is intentionally focused on schematic
    // emission, separate from mechanical-feature-rule coverage.)
}

/// Boundary-case coverage for the `(w.min(h) * 0.09).clamp(3.0, 8.0)` mounting-hole
/// inset formula at `mechanical_features`. Pinned to specific sizes so a contributor
/// silently widening the clamp range fails the test rather than emitting holes in
/// wrong places.
#[test]
fn mechanical_features_scales_with_board_and_clamps() {
    // Small board (10×10): 0.09·10 = 0.9 → clamp-lo to 3.0.
    let f_small = mechanical_features(10.0, 10.0);
    assert!(
        (f_small[0].x - 3.0).abs() < 1e-9 && (f_small[0].y - 3.0).abs() < 1e-9,
        "small-board clamp-low: {:?}",
        f_small[0]
    );
    // Mid board (50×50): 0.09·50 = 4.5 → no clamp, value passes through.
    let f_mid = mechanical_features(50.0, 50.0);
    assert!(
        (f_mid[0].x - 4.5).abs() < 1e-9 && (f_mid[0].y - 4.5).abs() < 1e-9,
        "mid-board no clamp: {:?}",
        f_mid[0]
    );
    // Large board (200×200): 0.09·200 = 18.0 → clamp-hi to 8.0.
    let f_large = mechanical_features(200.0, 200.0);
    assert!(
        (f_large[0].x - 8.0).abs() < 1e-9 && (f_large[0].y - 8.0).abs() < 1e-9,
        "large-board clamp-hi: {:?}",
        f_large[0]
    );
    // Sanity: 7 features every time (4 corner mounting holes + 3 fiducials).
    assert_eq!(f_small.len(), 7);
    assert_eq!(f_large.len(), 7);
}
