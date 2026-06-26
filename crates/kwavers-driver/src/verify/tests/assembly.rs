//! Physical assembly and keep-in verification tests.

use super::*;
use crate::board::{NetClassKind, Track};

#[test]
fn model_dims_parsed_from_kicad_filenames() {
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
    use crate::board::Board;
    let board = Board::new(spec());
    let fp = FootprintDef::new(
        "U",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![],
    );
    let lib = vec![fp];
    let edge = comp_at(0, "U_EDGE", 1.0, 10.0, None);
    let center = comp_at(0, "U_OK", 10.0, 10.0, None);
    let r = keepin(&board, &[edge, center], &lib, Nm::from_mm(0.5));
    assert_eq!(r.edge_violations.len(), 1);
    assert_eq!(r.edge_violations[0].0, "U_EDGE");
    assert!((r.edge_violations[0].1 - 1.5).abs() < 1e-6);
    assert!(!r.pass);
}

#[test]
fn keepin_flags_wide_track_inside_board_edge_clearance() {
    use crate::board::Board;
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
    assert!((r.copper_edge_violations[0].1 - 0.193).abs() < 1.0e-6);
    assert!(!r.pass);
}

#[test]
fn test_new_art_nasa_design_rules() {
    use crate::audit::audit;
    use crate::board::{Board, Via, Zone, ZoneFill};
    use crate::rules::DesignRules;

    let mut b = Board::new(spec());
    let rules = DesignRules::holohv();
    let net = b.add_net("TX_0_P", NetClassKind::Signal);
    let net_n = b.add_net("TX_0_N", NetClassKind::Signal);
    let other_net = b.add_net("OTHER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);

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
        pos: Point::new(Nm::from_mm(10.7), Nm::from_mm(10.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: other_net,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Micro,
        filled: false,
    });

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

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(0.4), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(0.4), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });

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
