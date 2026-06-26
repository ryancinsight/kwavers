//! EMI / proximity high-speed audit tests.

use super::super::board;
use crate::audit::{audit, emi_hotspots};
use crate::board::{Board, LayerId, NetClassKind, Pad, Track};
use crate::geom::{Nm, Point};
use crate::place::{Component, FootprintDef, PadDef, Placement, Role, Rot};
use crate::rules::DesignRules;

#[test]
fn emi_hotspots_flags_only_hv_to_lv_pairs() {
    let mut b = board();
    let hv = b.add_net("VPP", NetClassKind::Hv);
    let lv = b.add_net("CTRL", NetClassKind::Signal);
    let lv2 = b.add_net("CLK", NetClassKind::Signal);
    let pad = |b: &mut Board, x, y, n| {
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            layers: vec![LayerId(0)],
            net: Some(n),
        })
    };
    pad(&mut b, 5.0, 5.0, hv);
    pad(&mut b, 7.0, 5.0, lv); // 2 mm from HV ⇒ one HV↔LV hotspot (within 6 mm)
    pad(&mut b, 8.0, 5.0, lv2); // LV↔LV with the other LV ⇒ ignored; 3 mm from HV ⇒ also a hotspot
    pad(&mut b, 19.0, 19.0, lv); // far from HV ⇒ no hotspot
    let pts = emi_hotspots(&b, Nm::from_mm(6.0));
    // HV pad pairs with the two near LV pads (2 mm, 3 mm) ⇒ 2 hotspots; LV↔LV and the far LV none.
    assert_eq!(pts.len(), 2, "only HV↔LV pairs within 6 mm count");
}

#[test]
fn detects_crossing_flight_lines() {
    let mut b = board();
    let a = b.add_net("A", NetClassKind::Signal);
    let c = b.add_net("B", NetClassKind::Signal);
    let pad = |b: &mut Board, x, y, n| {
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            layers: vec![LayerId(0)],
            net: Some(n),
        })
    };
    // Net A: left→right across the middle. Net B: bottom→top across the middle. They cross.
    pad(&mut b, 0.0, 10.0, a);
    pad(&mut b, 20.0, 10.0, a);
    pad(&mut b, 10.0, 0.0, c);
    pad(&mut b, 10.0, 20.0, c);
    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.crossings, 1,
        "the two flight lines must cross exactly once"
    );
}

#[test]
fn detects_near_short_and_weights_hv() {
    let mut b = board();
    let hv = b.add_net("VPP", NetClassKind::Hv);
    let lv = b.add_net("CTRL", NetClassKind::Signal);
    // Two pads of different nets 0.2 mm apart (< 3*0.13 = 0.39 mm margin).
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(hv),
    });
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(5.2), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(lv),
    });
    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(r.near_shorts, 1, "the close HV/LV pad pair is a near-short");
    assert!(r.risk_score > 0.0);
}

#[test]
fn detects_high_speed_active_ic_near_board_edge() {
    let mut b = board();
    let tx = b.add_net("TX_CLK", NetClassKind::Signal);
    let ctrl = b.add_net("CTRL", NetClassKind::Signal);

    let lib = vec![
        FootprintDef::new(
            "ACTIVE",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "EDGE_CONN",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::Connector,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let comp = |fp, net, refdes: &str, x, y| Component {
        fp,
        nets: vec![Some(net)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![
        comp(0, tx, "U_EDGE", 3.0, 10.0),
        comp(0, tx, "U_CENTER", 10.0, 10.0),
        comp(0, ctrl, "U_CTRL_EDGE", 3.0, 15.0),
        comp(1, tx, "J_EDGE", 3.0, 3.0),
    ];

    let r = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        r.high_speed_component_edge_violations, 1,
        "only the active IC carrying a high-speed net inside the 3 mm edge keepout is flagged"
    );
}

#[test]
fn detects_high_speed_termination_far_from_active_ic() {
    let mut b = board();
    let tx = b.add_net("TX_TERM", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    let lib = vec![
        FootprintDef::new(
            "ACTIVE",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "R0402",
            (Nm::from_mm(1.2), Nm::from_mm(0.6)),
            Role::Passive,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.4), Nm(0)),
                    size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.4), Nm(0)),
                    size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        ),
    ];
    let comp = |fp, nets, refdes: &str, x, y| Component {
        fp,
        nets,
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![
        comp(0, vec![Some(tx)], "U1", 10.0, 10.0),
        comp(1, vec![Some(tx), Some(gnd)], "R_NEAR", 11.7, 10.0),
        comp(1, vec![Some(tx), Some(gnd)], "R_FAR", 17.0, 10.0),
        comp(1, vec![Some(tx), Some(gnd)], "C_IGNORE", 17.0, 12.0),
    ];

    let r = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        r.high_speed_termination_placement_violations, 1,
        "only the resistor-like high-speed terminator outside the 2 mm active-pad budget is flagged"
    );
}

#[test]
fn detects_unrelated_high_speed_parallel_spacing() {
    let mut b = board();
    let a = b.add_net("TX_A", NetClassKind::Signal);
    let c = b.add_net("TX_C", NetClassKind::Signal);
    for (net, y) in [(a, 4.0), (c, 4.55)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.high_speed_parallel_spacing_violations, 1,
        "unrelated long parallel TX traces closer than 3W are flagged"
    );

    let mut generic = board();
    let tx0 = generic.add_net("TX_0", NetClassKind::Signal);
    let tx1 = generic.add_net("TX_1", NetClassKind::Signal);
    for (net, y) in [(tx0, 4.0), (tx1, 5.0)] {
        generic.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let generic_report = audit(&generic, &[], &[], &DesignRules::holohv());
    assert_eq!(
        generic_report.high_speed_parallel_spacing_violations, 0,
        "a 0.85 mm edge gap clears generic 3W spacing for non-clock high-speed traces"
    );

    let mut clock = board();
    let clk = clock.add_net("TX_CLK", NetClassKind::Signal);
    let data = clock.add_net("TX_DATA", NetClassKind::Signal);
    for (net, y) in [(clk, 4.0), (data, 5.0)] {
        clock.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let clock_report = audit(&clock, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clock_report.high_speed_parallel_spacing_violations, 1,
        "the same 0.85 mm edge gap violates the 1.27 mm clock keepout"
    );

    let mut diff_pair = board();
    let p = diff_pair.add_net("LANE_P", NetClassKind::Signal);
    let n = diff_pair.add_net("LANE_N", NetClassKind::Signal);
    for (net, y) in [(p, 4.0), (n, 4.6)] {
        diff_pair.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let paired = audit(&diff_pair, &[], &[], &DesignRules::holohv());
    assert_eq!(
        paired.high_speed_parallel_spacing_violations, 0,
        "true P/N differential mates are exempt from unrelated-trace crosstalk spacing"
    );

    let mut short = board();
    let s0 = short.add_net("TX_S0", NetClassKind::Signal);
    let s1 = short.add_net("TX_S1", NetClassKind::Signal);
    for (net, y) in [(s0, 4.0), (s1, 4.6)] {
        short.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(4.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let short_overlap = audit(&short, &[], &[], &DesignRules::holohv());
    assert_eq!(
        short_overlap.high_speed_parallel_spacing_violations, 0,
        "short pad-entry adjacency below the coupled-length threshold is not counted"
    );
}

#[test]
fn detects_adjacent_layer_high_speed_parallelism() {
    let mut broadside = board();
    let tx0 = broadside.add_net("TX_TOP", NetClassKind::Signal);
    let tx1 = broadside.add_net("TX_INNER", NetClassKind::Signal);
    for (net, layer) in [(tx0, LayerId(0)), (tx1, LayerId(1))] {
        broadside.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(4.0)),
            width: Nm::from_mm(0.15),
            layer,
            net,
        });
    }
    let broadside_report = audit(&broadside, &[], &[], &DesignRules::holohv());
    assert_eq!(
        broadside_report.high_speed_parallel_spacing_violations, 0,
        "same-layer parallel spacing is not responsible for adjacent-layer coupling"
    );
    assert_eq!(
        broadside_report.high_speed_adjacent_layer_parallel_violations, 1,
        "overlapping adjacent-layer high-speed runs should be routed orthogonally or separated"
    );

    let mut orthogonal = board();
    let x = orthogonal.add_net("TX_X", NetClassKind::Signal);
    let y = orthogonal.add_net("TX_Y", NetClassKind::Signal);
    orthogonal.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
        end: Point::new(Nm::from_mm(10.0), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: x,
    });
    orthogonal.tracks.push(Track {
        start: Point::new(Nm::from_mm(6.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(6.0), Nm::from_mm(9.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: y,
    });
    let orthogonal_report = audit(&orthogonal, &[], &[], &DesignRules::holohv());
    assert_eq!(
        orthogonal_report.high_speed_adjacent_layer_parallel_violations, 0,
        "adjacent-layer high-speed crossings are orthogonal, not broadside parallel"
    );

    let mut separated = board();
    let s0 = separated.add_net("TX_S0", NetClassKind::Signal);
    let s1 = separated.add_net("TX_S1", NetClassKind::Signal);
    for (net, layer, y) in [(s0, LayerId(0), 4.0), (s1, LayerId(1), 5.0)] {
        separated.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer,
            net,
        });
    }
    let separated_report = audit(&separated, &[], &[], &DesignRules::holohv());
    assert_eq!(
        separated_report.high_speed_adjacent_layer_parallel_violations, 0,
        "0.85 mm planar edge offset clears the adjacent-layer 3W broadside budget"
    );
}
