//! Slice-wide tests for the [`verify`](super) facade.
//!
//! Collected from the previously-inline `#[cfg(test)] mod tests` block of `src/verify.rs`.
//! The shared helpers (`spec`, `two_pad_fp`, `comp`, `comp_at`) live here as siblings of the
//! per-axis production code; the parser test for [`super::assembly::model_dims_mm`] reaches the
//! helper through `super::assembly::*` (it's marked `pub(super)` solely so this test module can
//! exercise the implementation directly).
use super::*;
use crate::board::{Board, LayerId, NetClassKind, NetId, Track, Via, Zone, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::component::{Component, Placement};
use crate::place::footprint::{FootprintDef, PadDef, Role};
use crate::place::rotation::Rot;
use crate::rules::DesignRules;
use std::collections::HashMap;

mod lvs;

pub(super) fn spec() -> GridSpec {
    GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap()
}

pub(super) fn two_pad_fp() -> FootprintDef {
    FootprintDef::new(
        "R",
        (Nm::from_mm(2.0), Nm::from_mm(1.0)),
        Role::Passive,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )
    .with_model("m.step", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
}

pub(super) fn comp(fp: usize, refdes: &str, nets: Vec<Option<NetId>>) -> Component {
    Component {
        fp,
        nets,
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }
}

pub(super) fn comp_at(fp: usize, refdes: &str, x: f64, y: f64, assoc: Option<usize>) -> Component {
    Component {
        fp,
        nets: Vec::new(),
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: assoc,
        locked: false,
        ..Default::default()
    }
}

#[test]
fn model_dims_parsed_from_kicad_filenames() {
    // Reach the `pub(super)` [`super::assembly::model_dims_mm`] helper through the fully
    // qualified path — glob imports (`use super::*`) only pick up public symbols, so the
    // slice-internal helper that is only exposed so this test can drive the parser is reached
    // here by its qualified name within the same [`crate::verify`] tree.
    use crate::verify::assembly::model_dims_mm;
    assert_eq!(
        model_dims_mm("Package_QFP.3dshapes/LQFP-100_14x14mm_P0.5mm.step"),
        Some((14.0, 14.0))
    );
    assert_eq!(
        model_dims_mm("Oscillator_SMD_EuroQuartz_XO53-4Pin_5.0x3.2mm.step"),
        Some((5.0, 3.2))
    );
    assert_eq!(
        model_dims_mm("JEITA_SOIC-8_3.9x4.9mm_P1.27mm.step"),
        Some((3.9, 4.9))
    );
    assert_eq!(model_dims_mm("no_size_here.step"), None);
}

#[test]
fn assembly_flags_a_model_larger_than_its_courtyard() {
    // Courtyard 3.2×2.5, model 5.0×3.2 — the body does not fit either orientation ⇒ flagged.
    let small = FootprintDef::new(
        "OSC",
        (Nm::from_mm(3.2), Nm::from_mm(2.5)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )
    .with_model(
        "Oscillator_5.0x3.2mm.step",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    );
    let pads = small.pads.clone();
    let r = assembly(&[], &[small], Nm::from_mm(0.5));
    assert_eq!(r.oversized_models.len(), 1, "oversized model is flagged");
    assert!(!r.pass);
    // Enlarging the courtyard to enclose the model clears it.
    let ok = FootprintDef::new(
        "OSC",
        (Nm::from_mm(5.6), Nm::from_mm(3.8)),
        Role::ActiveIc,
        pads,
    )
    .with_model(
        "Oscillator_5.0x3.2mm.step",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    );
    assert!(assembly(&[], &[ok], Nm::from_mm(0.5)).pass);
}

#[test]
fn assembly_flags_bottom_side_smd_and_non_top_through_hole() {
    let fp = |name: &str, layers: Vec<LayerId>| {
        FootprintDef::new(
            name,
            (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Role::Passive,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers,
                power_pin: false,
            }],
        )
    };
    let component = |fp: usize, refdes: &str| Component {
        fp,
        nets: vec![None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(fp as f64 * 3.0), Nm::from_mm(2.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    let clean_lib = vec![fp("TOP_SMD", vec![LayerId(0)])];
    let clean = assembly(&[component(0, "R1")], &clean_lib, Nm::from_mm(0.5));
    assert!(
        clean.pass,
        "top-side SMD component satisfies the assembly-side rule"
    );

    let dirty_lib = vec![
        fp("TOP_SMD", vec![LayerId(0)]),
        fp("BOTTOM_SMD", vec![LayerId(1)]),
        fp("INNER_TH", vec![LayerId(1), LayerId(2)]),
    ];
    let dirty = assembly(
        &[component(0, "R1"), component(1, "C1"), component(2, "J1")],
        &dirty_lib,
        Nm::from_mm(0.5),
    );
    assert!(
        !dirty.pass,
        "bottom-side SMD and non-top through-hole placement are assembly faults"
    );
    assert_eq!(
        dirty.side_violations.len(),
        2,
        "only C1 and J1 violate the side-placement policy"
    );
    assert_eq!(dirty.side_violations[0].refdes, "C1");
    assert_eq!(
        dirty.side_violations[0].reason,
        "SMD pads must be on the top assembly side"
    );
    assert_eq!(dirty.side_violations[1].refdes, "J1");
    assert_eq!(
        dirty.side_violations[1].reason,
        "through-hole pads must include the top assembly side"
    );
}

#[test]
fn keepin_flags_a_component_over_the_board_edge() {
    // 20×20 mm board, 0.5 mm margin. A 4×4 courtyard centred at x=1 reaches x=−1 ⇒ overhang.
    let board = Board::new(spec());
    let fp = FootprintDef::new(
        "U",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![],
    );
    let lib = vec![fp];
    let edge = comp_at(0, "U_EDGE", 1.0, 10.0, None); // courtyard x ∈ [−1, 3] ⇒ crosses keep-in
    let center = comp_at(0, "U_OK", 10.0, 10.0, None);
    let r = keepin(&board, &[edge, center], &lib, Nm::from_mm(0.5));
    assert_eq!(r.edge_violations.len(), 1);
    assert_eq!(r.edge_violations[0].0, "U_EDGE");
    // Overhang = margin(0.5) − courtyard.min.x(−1) = 1.5 mm.
    assert!((r.edge_violations[0].1 - 1.5).abs() < 1e-6);
    assert!(!r.pass);
}

#[test]
fn keepin_flags_wide_track_inside_board_edge_clearance() {
    let mut board = Board::new(spec());
    let vpp = board.add_net("VPP", NetClassKind::Hv);
    board.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(0.5)),
        end: Point::new(Nm::from_mm(10.0), Nm::from_mm(0.5)),
        width: Nm::from_mm(0.386),
        layer: LayerId(0),
        net: vpp,
    });
    let r = keepin(&board, &[], &[], Nm::from_mm(0.5));
    assert!(r.edge_violations.is_empty());
    assert_eq!(r.copper_edge_violations.len(), 1);
    assert_eq!(r.copper_edge_violations[0].0, "track[0]");
    // Copper clearance = 0.500 - 0.386/2 = 0.307 mm, so shortfall is 0.193 mm.
    assert!((r.copper_edge_violations[0].1 - 0.193).abs() < 1.0e-6);
    assert!(!r.pass);
}

#[test]
fn decoupling_proximity_measures_to_the_nearest_power_pin() {
    // IC with a single *power* pad offset 2 mm left of centre (so at x=8 when the IC is at x=10).
    let ic = FootprintDef::new(
        "U",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
            size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    );
    let cap = FootprintDef::new(
        "C",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![],
    );
    let lib = vec![ic, cap];
    // IC at (10,10) ⇒ power pin at (8,10). NEAR cap at (10,10) is 2 mm from the pin (passes a 3 mm
    // budget); FAR cap at (13,10) is 5 mm from the pin (fails). Both are >3 mm from the IC
    // *centre*, so a centre-based check would wrongly flag the near one too.
    let comps = vec![
        comp_at(0, "U1", 10.0, 10.0, None),
        comp_at(1, "C_NEAR", 10.0, 10.0, Some(0)),
        comp_at(1, "C_FAR", 13.0, 10.0, Some(0)),
    ];
    let r = decoupling_proximity(&comps, &lib, 3.0);
    assert_eq!(
        r.far_caps.len(),
        1,
        "only the far cap fails (measured to the power pin)"
    );
    assert_eq!(r.far_caps[0].0, "C_FAR");
    assert!((r.far_caps[0].1 - 5.0).abs() < 1e-6);
    assert!(!r.pass);
}

#[test]
fn erc_flags_unconnected_pad_and_floating_net() {
    let mut b = Board::new(spec());
    let a = b.add_net("A", NetClassKind::Signal);
    let lib = vec![two_pad_fp()];
    // Pad 0 → net A (only terminal ⇒ floating), pad 1 → no net (unconnected).
    let comps = vec![comp(0, "R1", vec![Some(a), None])];
    let symbol_pin_maps = HashMap::new();
    let r = erc(&b, &comps, &lib, &symbol_pin_maps);
    assert_eq!(r.unconnected_pads, vec![("R1".into(), 1)]);
    assert_eq!(r.floating_nets, vec!["A".to_string()]);
    assert!(!r.pass);
}

#[test]
fn erc_flags_power_pin_on_signal_net() {
    let mut b = Board::new(spec());
    let sig = b.add_net("CTRL", NetClassKind::Signal);
    let mut fp = two_pad_fp();
    fp.pads[0].power_pin = true; // a power pin...
    let lib = vec![fp];
    let comps = vec![comp(0, "U1", vec![Some(sig), Some(sig)])];
    let symbol_pin_maps = HashMap::new();
    let r = erc(&b, &comps, &lib, &symbol_pin_maps);
    assert_eq!(r.power_pin_on_signal, vec![("U1".into(), 0, "CTRL".into())]);
    assert!(!r.pass);
}

#[test]
fn bom_flags_duplicate_refdes() {
    let lib = vec![two_pad_fp()];
    let comps = vec![
        comp(0, "R1", vec![None, None]),
        comp(0, "R1", vec![None, None]), // duplicate
        comp(0, "R2", vec![None, None]),
    ];
    let r = bom(&comps, &lib);
    assert_eq!(r.part_count, 3);
    assert_eq!(r.duplicate_refdes, vec!["R1".to_string()]);
    assert!(!r.pass);
}

#[test]
fn schematic_isolation_bfs_detects_leakage() {
    let mut b = Board::new(spec());
    let sclk = b.add_net("BUS_SCLK", NetClassKind::Signal);
    let sclk_p = b.add_net("SCLK_P", NetClassKind::Signal);

    let lib = vec![two_pad_fp()];
    // Case 1: isolated. sclk (control) on ISO1, sclk_p (pulser) on ISO1. They are separate nets and ISO1 isolates them.
    let comps_clean = vec![
        comp(0, "J_STACK", vec![Some(sclk)]),
        comp(0, "ISO1", vec![Some(sclk), Some(sclk_p)]),
        comp(0, "U1", vec![Some(sclk_p)]),
    ];
    let r_clean = schematic_isolation_bfs(&b, &comps_clean, &lib);
    assert!(r_clean.pass);

    // Case 2: bridged by a resistor R_ERR.
    let comps_dirty = vec![
        comp(0, "J_STACK", vec![Some(sclk)]),
        comp(0, "ISO1", vec![Some(sclk), Some(sclk_p)]),
        comp(0, "R_ERR", vec![Some(sclk), Some(sclk_p)]), // resistor bypasses isolation!
        comp(0, "U1", vec![Some(sclk_p)]),
    ];
    let r_dirty = schematic_isolation_bfs(&b, &comps_dirty, &lib);
    assert!(!r_dirty.pass);
    assert_eq!(r_dirty.violations.len(), 1);
    assert_eq!(r_dirty.violations[0].control_net, "BUS_SCLK");
    assert_eq!(r_dirty.violations[0].hv_net, "SCLK_P");
    assert_eq!(
        r_dirty.violations[0].path,
        vec![
            "BUS_SCLK".to_string(),
            "R_ERR".to_string(),
            "SCLK_P".to_string()
        ]
    );
}

#[test]
fn ac_coupling_flags_low_impedance() {
    let mut b = Board::new(spec());
    let trig = b.add_net("TRIG_0", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    // A parallel run of TRIG_0 (switching) and GND with very narrow spacing.
    // Spacing: center-to-center = 0.5 mm. Widths = 0.2 mm. Spacing = 0.5 - 0.2 = 0.3 mm.
    // Overlap: 50 mm.
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(51.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: trig,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.5)),
        end: Point::new(Nm::from_mm(51.0), Nm::from_mm(5.5)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: gnd,
    });

    let r = parasitic_ac_coupling_check(&b);
    // Capacitance = 50 * 0.008 * ln(1 + 2*0.2 / 0.3) = 0.4 * ln(1 + 1.333) = 0.4 * ln(2.333) = 0.4 * 0.847 = 0.339 pF
    // Impedance = 13642.0 / 0.339 = 40.2 kohm (above 1 kohm limit) -> passes.
    assert!(r.pass);

    // Now make the run extremely long (e.g. 3000 mm) to trigger a low impedance (< 1000 Ohm) violation
    b.tracks.clear();
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: trig,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.5)),
        end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(5.5)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: gnd,
    });

    let r2 = parasitic_ac_coupling_check(&b);
    assert!(!r2.pass);
    assert_eq!(r2.violations.len(), 1);
    assert_eq!(r2.violations[0].switching_net, "TRIG_0");
}

#[test]
fn test_new_art_nasa_design_rules() {
    use crate::audit::audit;

    let mut b = Board::new(spec());
    let rules = DesignRules::holohv();
    let net = b.add_net("TX_0_P", NetClassKind::Signal);
    let net_n = b.add_net("TX_0_N", NetClassKind::Signal);
    let other_net = b.add_net("OTHER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    // 1. Sharp Bends Test: horizontal track meeting vertical track of same net
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });

    // 2. Serpentine spacing (4W) and length (1.5W)
    // Add one short 0.1 mm segment (less than 1.5*0.15 = 0.225 mm) connected
    // collinearly at both ends so this case isolates segment length rather than adding bends.
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(5.1), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.1), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(6.1), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    // Parallel tracks of same net closer than 4W (4 * 0.15 = 0.6 mm)
    // Track A at y=5.0, Track B at y=5.5 (0.5 mm difference, < 0.6 mm)
    // Overlapping x-range [1.0, 4.0]
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: net_n,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(5.5)),
        end: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.5)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: net_n,
    });

    // 3. Via-to-via spacing: two vias of different nets spaced 0.3 mm pad-to-pad (limit = 0.381 mm)
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Micro,
        filled: false,
    });
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(10.7), Nm::from_mm(10.0)), // distance = 0.7 mm. gap = 0.7 - 0.46 = 0.24 mm (< 0.381 mm)
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: other_net,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Micro,
        filled: false,
    });

    // 4. Differential pair corridor violation: other net pad/via between TX_0_P and TX_0_N parallel tracks
    // TX_0_P: [12.0, 15.0] at y=12.0
    // TX_0_N: [12.0, 15.0] at y=13.0
    // Other net pad at x=13.5, y=12.5 (exactly between them)
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(12.0), Nm::from_mm(12.0)),
        end: Point::new(Nm::from_mm(15.0), Nm::from_mm(12.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(12.0), Nm::from_mm(13.0)),
        end: Point::new(Nm::from_mm(15.0), Nm::from_mm(13.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: net_n,
    });
    b.add_pad(crate::board::Pad {
        pos: Point::new(Nm::from_mm(13.5), Nm::from_mm(12.5)),
        layers: vec![LayerId(0)],
        net: Some(other_net),
    });

    // 5. High-speed edge violation: TX_0_P (high speed) routed at x=0.4 mm (< 1.0 mm clearance)
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(0.4), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(0.4), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });

    // 6. Split plane crossing: high-speed track crossing zone boundary
    // Zone covering x in [0, 10], y in [0, 20]
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(0),
        polygon: vec![
            Point::new(Nm(0), Nm(0)),
            Point::new(Nm::from_mm(10.0), Nm(0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(20.0)),
            Point::new(Nm(0), Nm::from_mm(20.0)),
        ],
        fill: ZoneFill::Solid,
    });
    // High speed track from x=9.0 to x=11.0 (crosses zone boundary at x=10.0)
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(9.0), Nm::from_mm(15.0)),
        end: Point::new(Nm::from_mm(11.0), Nm::from_mm(15.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });

    let r = audit(&b, &[], &[], &rules);
    assert_eq!(r.sharp_bends, 1, "90 degree bend should be flagged");
    assert_eq!(
        r.serpentine_spacing_violations, 1,
        "4W serpentine spacing violation should be flagged"
    );
    assert_eq!(
        r.serpentine_length_violations, 1,
        "1.5W segment length violation should be flagged"
    );
    assert_eq!(
        r.via_spacing_violations, 1,
        "via spacing violation should be flagged"
    );
    assert_eq!(
        r.diff_pair_violations, 1,
        "other net pad inside diff pair corridor should be flagged"
    );
    assert_eq!(
        r.diff_pair_length_mismatch_violations, 1,
        "differential pair members with unmatched routed length should be flagged"
    );
    assert_eq!(
        r.high_speed_edge_violations, 1,
        "high speed edge violation should be flagged"
    );
    assert_eq!(
        r.high_speed_parallel_spacing_violations, 0,
        "differential pair mates are not unrelated high-speed parallel spacing violations"
    );
    assert_eq!(
        r.reference_plane_margin_violations, 1,
        "high speed trace inside a reference zone must keep the 3W plane margin"
    );
    assert_eq!(
        r.high_speed_transition_ground_via_violations, 1,
        "high speed layer transition must have a local ground transition via"
    );
    assert_eq!(
        r.split_plane_crossings, 1,
        "split plane crossing should be flagged"
    );
}
