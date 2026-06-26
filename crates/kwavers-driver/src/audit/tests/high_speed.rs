use super::*;

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
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

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
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

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
fn detects_signal_track_intruding_on_reference_plane() {
    use crate::board::{Zone, ZoneFill};

    let mut b = board();
    let tx = b.add_net("TX_PLANE_CUT", NetClassKind::Signal);
    let ctrl = b.add_net("CTRL_OTHER_LAYER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: tx,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(12.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(12.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: ctrl,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(14.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(14.0)),
        width: Nm::from_mm(0.25),
        layer: LayerId(1),
        net: gnd,
    });

    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.reference_plane_intrusion_violations, 1,
        "only the non-plane signal track routed through the reference-plane zone is flagged"
    );
}

#[test]
fn detects_signal_crossing_opposite_split_ground_domain() {
    use crate::board::{Zone, ZoneFill};

    let mut b = board();
    let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
    let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
    let agnd = b.add_net("AGND", NetClassKind::Ground);
    let dgnd = b.add_net("DGND", NetClassKind::Ground);
    let zone = |net, x0: f64, x1: f64| Zone {
        net,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(x0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(x0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };
    b.zones.push(zone(agnd, 2.0, 10.0));
    b.zones.push(zone(dgnd, 10.0, 18.0));
    let route = |board: &mut Board, net, x0: f64, x1: f64, y: f64| {
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net,
        });
    };

    let mut clean = b.clone();
    route(&mut clean, analog, 3.0, 9.0, 6.0);
    route(&mut clean, digital, 11.0, 17.0, 8.0);
    let same_domain = audit(&clean, &[], &[], &DesignRules::holohv());
    assert_eq!(
        same_domain.split_domain_reference_violations, 0,
        "analog over AGND and digital over DGND stay inside their split-ground domains"
    );

    let mut crossed = b;
    route(&mut crossed, digital, 3.0, 9.0, 6.0);
    route(&mut crossed, analog, 11.0, 17.0, 8.0);
    let opposite_domain = audit(&crossed, &[], &[], &DesignRules::holohv());
    assert_eq!(
        opposite_domain.split_domain_reference_violations, 2,
        "digital over AGND and analog over DGND both violate split-domain reference routing"
    );
}

#[test]
fn detects_mixed_domain_shared_ground_return_overlap() {
    let mut b = board();
    let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
    let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let route = |board: &mut Board, net, y: f64| {
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(3.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(17.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    };

    let mut separated = b.clone();
    route(&mut separated, analog, 6.0);
    route(&mut separated, digital, 8.0);
    let clean = audit(&separated, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.mixed_domain_shared_reference_violations, 0,
        "analog and digital returns sharing GND but separated by more than the sensitive keepout are clean"
    );

    route(&mut b, analog, 6.0);
    route(&mut b, digital, 6.8);
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.mixed_domain_shared_reference_violations, 1,
        "analog and digital tracks closer than the sensitive keepout over one GND plane overlap return-current corridors"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "mixed-domain return-current overlap must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_virtual_split_domain_crossing() {
    let mut b = board();
    let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
    let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
    for (net, x) in [(analog, 4.0), (digital, 16.0)] {
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(4.0)),
            layers: vec![LayerId(0)],
            net: Some(net),
        });
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(16.0)),
            layers: vec![LayerId(0)],
            net: Some(net),
        });
    }
    let route = |board: &mut Board, net, x0: f64, x1: f64, y: f64| {
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    };

    let mut clean_board = b.clone();
    route(&mut clean_board, analog, 3.0, 8.0, 6.0);
    route(&mut clean_board, digital, 12.0, 17.0, 14.0);
    let clean = audit(&clean_board, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.virtual_split_crossing_violations, 0,
        "analog and digital tracks staying on their centroid-derived virtual sides are clean"
    );

    route(&mut b, analog, 3.0, 12.0, 6.0);
    route(&mut b, digital, 12.0, 17.0, 14.0);
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.virtual_split_crossing_violations, 1,
        "the analog track crossing the inferred x=10 mm virtual split line is flagged"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "virtual split crossing must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_fragmented_same_net_ground_plane() {
    let mut b = board();
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let island = |x0: f64, x1: f64| Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(x0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(x0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };

    b.zones.push(island(2.0, 18.0));
    let continuous = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        continuous.ground_plane_fragmentation_violations, 0,
        "one ground pour on a layer is a continuous reference plane"
    );

    b.zones.push(island(20.0, 24.0));
    let fragmented = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        fragmented.ground_plane_fragmentation_violations, 1,
        "two same-net ground pour islands on one layer fragment the reference plane"
    );

    b.zones[1].fill = ZoneFill::Solid;
    let teardrop_like = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        teardrop_like.ground_plane_fragmentation_violations, 0,
        "solid same-net reinforcement zones are not counted as plane-pour fragmentation"
    );
}

#[test]
fn detects_high_speed_track_without_adjacent_reference_plane() {
    use crate::board::{Zone, ZoneFill};

    let mut b = board();
    let tx = b.add_net("TX_REF", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx,
    });

    let missing = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        missing.reference_plane_absence_violations, 1,
        "a high-speed segment with no adjacent reference zone is flagged"
    );

    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(0),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let same_layer = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        same_layer.reference_plane_absence_violations, 1,
        "same-layer copper is not an adjacent reference plane"
    );

    b.zones.clear();
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let partial = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        partial.reference_plane_absence_violations, 1,
        "adjacent reference coverage must cover the segment start, middle, and end"
    );

    b.zones[0].polygon = vec![
        Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
        Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
        Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
        Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
    ];
    let covered = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        covered.reference_plane_absence_violations, 0,
        "a full adjacent ground zone supplies the high-speed reference plane"
    );
}

#[test]
fn detects_inner_high_speed_track_without_dual_ground_reference() {
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
    let tx = b.add_net("TX_INNER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: tx,
    });
    let reference = |layer| Zone {
        net: gnd,
        layer,
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };
    b.zones.push(reference(LayerId(0)));
    let one_sided = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        one_sided.reference_plane_absence_violations, 0,
        "one adjacent ground zone satisfies the generic adjacent-reference rule"
    );
    assert_eq!(
        one_sided.inner_layer_dual_ground_reference_violations, 1,
        "an inner high-speed layer requires ground reference zones on both adjacent layers"
    );

    b.zones.push(reference(LayerId(2)));
    let dual = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dual.inner_layer_dual_ground_reference_violations, 0,
        "ground zones on both adjacent layers clear the inner-layer high-speed reference rule"
    );
}

#[test]
fn detects_power_plane_reference_without_stitching_caps() {
    use crate::board::{Zone, ZoneFill};
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

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
    use crate::board::{Zone, ZoneFill};
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

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
    use crate::board::{Zone, ZoneFill};
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

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
    use crate::board::{Via, ViaKind};

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
fn detects_unfilled_via_in_non_ground_smd_pad() {
    use crate::board::{Via, ViaKind};

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
