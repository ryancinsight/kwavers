//! Stitching / transition high-speed audit tests.

use super::super::board;
use crate::audit::audit;
use crate::board::{Board, LayerId, NetClassKind, NetId, Pad, Track, Via, ViaKind, Zone, ZoneFill};
use crate::geom::{Nm, Point};
use crate::place::{Component, FootprintDef, PadDef, Placement, Role, Rot};
use crate::rules::DesignRules;

#[test]
fn detects_power_plane_reference_without_stitching_caps() {
    let mut b = board();
    let tx = b.add_net("TX_PWR_REF", NetClassKind::Signal);
    let pwr = b.add_net("VDD", NetClassKind::Power);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx,
    });
    b.zones.push(Zone {
        net: pwr,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let cap_fp = FootprintDef::new(
        "C0402",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.3), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.3), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let mk_cap = |refdes: &str, x: f64| Component {
        fp: 0,
        nets: vec![Some(pwr), Some(gnd)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let lib = vec![cap_fp];

    let missing = audit(&b, &[], &lib, &DesignRules::holohv());
    assert_eq!(
        missing.power_reference_stitching_cap_violations, 1,
        "a power-plane-referenced high-speed track needs endpoint stitching capacitors"
    );

    let one_ended = audit(&b, &[mk_cap("C_SRC", 4.5)], &lib, &DesignRules::holohv());
    assert_eq!(
        one_ended.power_reference_stitching_cap_violations, 1,
        "stitching only the source endpoint is insufficient"
    );

    let stitched = audit(
        &b,
        &[mk_cap("C_SRC", 4.5), mk_cap("C_SINK", 15.5)],
        &lib,
        &DesignRules::holohv(),
    );
    assert_eq!(
        stitched.power_reference_stitching_cap_violations, 0,
        "source and sink stitching capacitors clear the power-reference requirement"
    );

    b.zones[0].net = gnd;
    let ground_referenced = audit(&b, &[], &lib, &DesignRules::holohv());
    assert_eq!(
        ground_referenced.power_reference_stitching_cap_violations, 0,
        "a ground-plane reference does not require power-reference stitching capacitors"
    );
}

#[test]
fn detects_asymmetric_diff_pair_power_reference_stitching_caps() {
    let mut b = board();
    let p = b.add_net("MGT_P", NetClassKind::Signal);
    let n = b.add_net("MGT_N", NetClassKind::Signal);
    let vref = b.add_net("VREF", NetClassKind::Power);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    b.zones.push(Zone {
        net: vref,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            Point::new(Nm::from_mm(14.0), Nm::from_mm(0.0)),
            Point::new(Nm::from_mm(14.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(0.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    for (net, y) in [(p, 4.0), (n, 5.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let cap_fp = FootprintDef::new(
        "C0402",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.25), Nm(0)),
                size: (Nm::from_mm(0.25), Nm::from_mm(0.25)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.25), Nm(0)),
                size: (Nm::from_mm(0.25), Nm::from_mm(0.25)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let cap = |refdes: &str, x, y| Component {
        fp: 0,
        nets: vec![Some(vref), Some(gnd)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let lib = vec![cap_fp];
    let matched = vec![
        cap("C_PS0", 2.0, 3.7),
        cap("C_PS1", 10.0, 3.7),
        cap("C_NS0", 2.0, 5.3),
        cap("C_NS1", 10.0, 5.3),
    ];
    let clean = audit(&b, &matched, &lib, &DesignRules::holohv());
    assert_eq!(
        clean.power_reference_stitching_cap_violations, 0,
        "each power-referenced pair endpoint has a local stitching capacitor"
    );
    assert_eq!(
        clean.diff_pair_stitching_cap_symmetry_violations, 0,
        "P/N stitching capacitors at matching stations are symmetric"
    );

    let shifted = vec![
        cap("C_PS0", 2.0, 3.7),
        cap("C_PS1", 10.0, 3.7),
        cap("C_NS0", 2.8, 5.3),
        cap("C_NS1", 10.8, 5.3),
    ];
    let dirty = audit(&b, &shifted, &lib, &DesignRules::holohv());
    assert_eq!(
        dirty.power_reference_stitching_cap_violations, 0,
        "shifted capacitors are still local to the signal endpoints"
    );
    assert_eq!(
        dirty.diff_pair_stitching_cap_symmetry_violations, 1,
        "0.8 mm P/N stitching-cap station mismatch exceeds the 0.5 mm symmetry tolerance"
    );
}

#[test]
fn split_plane_stitching_cap_must_be_local_to_crossing() {
    let mut b = board();
    let tx = b.add_net("TX_SPLIT", NetClassKind::Signal);
    let pwr = b.add_net("VDD", NetClassKind::Power);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(0),
        polygon: vec![
            Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(0.0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(20.0)),
            Point::new(Nm::from_mm(0.0), Nm::from_mm(20.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(9.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(11.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx,
    });

    let cap_fp = FootprintDef::new(
        "C0402",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.3), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.3), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let cap_at = |x: f64, y: f64| Component {
        fp: 0,
        nets: vec![Some(pwr), Some(gnd)],
        refdes: "C_SPLIT".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let lib = vec![cap_fp];

    let far = audit(&b, &[cap_at(14.0, 10.0)], &lib, &DesignRules::holohv());
    assert_eq!(
        far.split_plane_crossings, 1,
        "a stitching capacitor 4 mm from the split crossing is outside the 2 mm path budget"
    );

    let near = audit(&b, &[cap_at(10.0, 11.0)], &lib, &DesignRules::holohv());
    assert_eq!(
        near.split_plane_crossings, 0,
        "a stitching capacitor within 1 mm of the split crossing supplies a local return path"
    );

    let local_signal_cap = Component {
        nets: vec![Some(tx), Some(gnd)],
        ..cap_at(10.0, 11.0)
    };
    let not_stitched = audit(&b, &[local_signal_cap], &lib, &DesignRules::holohv());
    assert_eq!(
        not_stitched.split_plane_crossings, 1,
        "a local capacitor must bridge the crossed reference zone to another reference net"
    );
}

#[test]
fn detects_asymmetric_diff_pair_transition_ground_vias() {
    let mut b = board();
    let p = b.add_net("LANE_P", NetClassKind::Signal);
    let n = b.add_net("LANE_N", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    for (net, y) in [(p, 4.0), (n, 5.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let via_at = |net, x: f64, y: f64| Via {
        pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    };
    b.vias.push(via_at(p, 6.0, 4.0));
    b.vias.push(via_at(n, 6.0, 5.0));
    b.vias.push(via_at(gnd, 6.0, 3.7));
    b.vias.push(via_at(gnd, 6.0, 5.3));

    let matched = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        matched.high_speed_transition_ground_via_violations, 0,
        "both P/N layer transitions have local ground transition vias"
    );
    assert_eq!(
        matched.diff_pair_transition_ground_via_symmetry_violations, 0,
        "P/N ground transition vias at the same pair-axis station are symmetric"
    );

    b.vias[3].pos = Point::new(Nm::from_mm(7.0), Nm::from_mm(5.3));
    let shifted = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        shifted.high_speed_transition_ground_via_violations, 0,
        "the shifted ground via is still within the local return-via distance"
    );
    assert_eq!(
        shifted.diff_pair_transition_ground_via_symmetry_violations, 1,
        "a 1 mm P/N ground-via station mismatch exceeds the 0.5 mm symmetry tolerance"
    );
}

#[test]
fn detects_high_speed_transition_without_ground_via() {
    let mut b = board();
    let tx = b.add_net("TX_LAYER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(6.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: tx,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.high_speed_transition_ground_via_violations, 1,
        "a high-speed layer transition without a local ground transition via is flagged"
    );

    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(7.0), Nm::from_mm(6.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: gnd,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.high_speed_transition_ground_via_violations, 0,
        "a nearby ground transition via supplies the local return path"
    );
}

#[test]
fn detects_high_speed_terminal_without_ground_return() {
    let mut b = board();
    let tx = b.add_net("TX_TERM", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let p0 = Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0));
    let p1 = Point::new(Nm::from_mm(12.0), Nm::from_mm(4.0));
    for p in [p0, p1] {
        b.add_pad(Pad {
            pos: p,
            layers: vec![LayerId(0)],
            net: Some(tx),
        });
    }

    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.high_speed_terminal_ground_via_violations, 2,
        "both high-speed source/sink pads lack local ground return copper"
    );

    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(4.0)),
        layers: vec![LayerId(0)],
        net: Some(gnd),
    });
    let one_sided = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        one_sided.high_speed_terminal_ground_via_violations, 1,
        "only the terminal with a nearby ground feature is cleared"
    );

    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(12.0), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(gnd),
    });
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.high_speed_terminal_ground_via_violations, 0,
        "source and sink terminals both have local return copper"
    );
}

#[test]
fn detects_high_speed_via_far_from_same_net_pad() {
    let mut b = board();
    let tx = b.add_net("TX_PAD_VIA", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(tx),
    });
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(6.0)),
        layers: vec![LayerId(0)],
        net: Some(gnd),
    });

    let via_at = |board: &mut Board, net: NetId, x_mm: f64, y_mm: f64| {
        board.vias.push(Via {
            pos: Point::new(Nm::from_mm(x_mm), Nm::from_mm(y_mm)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
    };

    via_at(&mut b, tx, 6.5, 5.0);
    via_at(&mut b, gnd, 6.5, 6.0);
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.high_speed_via_pad_proximity_violations, 0,
        "a high-speed via 1.5 mm from its same-net pad stays inside the 2 mm budget"
    );
    assert_eq!(
        clean.high_speed_transition_ground_via_violations, 0,
        "the focused test keeps the high-speed transition return path locally satisfied"
    );

    b.vias.clear();
    via_at(&mut b, tx, 8.0, 5.0);
    via_at(&mut b, gnd, 8.0, 6.0);
    let far = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        far.high_speed_via_pad_proximity_violations, 1,
        "a high-speed via 3.0 mm from its same-net pad is outside the 2 mm budget"
    );
    assert_eq!(
        far.high_speed_transition_ground_via_violations, 0,
        "the violation is via-to-pad placement, not a missing ground transition via"
    );
}

#[test]
fn detects_same_net_non_ground_via_plane_hotspot_spacing() {
    let mut b = board();
    let tx = b.add_net("TX_CLUSTER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let via_at = |net, x| Via {
        pos: Point::new(Nm::from_mm(x), Nm::from_mm(8.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    };

    b.vias.push(via_at(tx, 5.0));
    b.vias.push(via_at(tx, 5.7));
    let clustered = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clustered.via_spacing_violations, 0,
        "same-net vias are outside the different-net via-spacing DRC"
    );
    assert_eq!(
        clustered.plane_hotspot_via_spacing_violations, 1,
        "0.24 mm outer gap violates the 15 mil plane-hotspot spacing budget"
    );

    b.vias[1].pos = Point::new(Nm::from_mm(6.0), Nm::from_mm(8.0));
    let spaced = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        spaced.plane_hotspot_via_spacing_violations, 0,
        "0.54 mm outer gap clears the 15 mil plane-hotspot spacing budget"
    );

    b.vias.clear();
    b.vias.push(via_at(gnd, 5.0));
    b.vias.push(via_at(gnd, 5.7));
    let ground_stitching = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        ground_stitching.plane_hotspot_via_spacing_violations, 0,
        "close same-net ground stitching vias are intentional return-plane ties"
    );
}

#[test]
fn detects_unfilled_via_in_non_ground_smd_pad() {
    let mut b = board();
    let sig = b.add_net("BGA_SIG", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let smd = Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0));
    let thermal = Point::new(Nm::from_mm(8.0), Nm::from_mm(5.0));
    let drilled = Point::new(Nm::from_mm(11.0), Nm::from_mm(5.0));
    b.add_pad(Pad {
        pos: smd,
        layers: vec![LayerId(0)],
        net: Some(sig),
    });
    b.add_pad(Pad {
        pos: thermal,
        layers: vec![LayerId(0)],
        net: Some(gnd),
    });
    b.add_pad(Pad {
        pos: drilled,
        layers: vec![LayerId(0), LayerId(1)],
        net: Some(sig),
    });
    let via = |net, pos, filled| Via {
        pos,
        drill: Nm::from_mm(0.1),
        diameter: Nm::from_mm(0.25),
        net,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled,
    };

    b.vias.push(via(sig, smd, false));
    b.vias.push(via(gnd, thermal, false));
    b.vias.push(via(sig, drilled, false));
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.unfilled_via_in_pad_violations, 1,
        "only the unfilled via inside the non-ground SMD pad violates VIPPO filling"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "unfilled via-in-pad must reject optimizer clean-board selection"
    );

    b.vias[0].filled = true;
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.unfilled_via_in_pad_violations, 0,
        "filled VIPPO plus unfilled ground thermal-pad/drilled-pad vias are accepted"
    );
}
