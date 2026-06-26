use super::*;

#[test]
fn detects_differential_pair_layer_and_via_mismatch() {
    use crate::board::{Via, ViaKind};

    let mut b = board();
    let p = b.add_net("CLK_P", NetClassKind::Signal);
    let n = b.add_net("CLK_N", NetClassKind::Signal);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: p,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: n,
    });
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(4.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: p,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });

    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_layer_mismatch_violations, 1,
        "pair members routed on different layer sets must be flagged"
    );
    assert_eq!(
        r.diff_pair_via_count_violations, 1,
        "pair members with different via counts must be flagged"
    );
}

#[test]
fn detects_same_interface_diff_pair_layer_mismatch() {
    use crate::board::{Via, ViaKind};

    let mut clean = board();
    let d0p = clean.add_net("MIPI_D0_P", NetClassKind::Signal);
    let d0n = clean.add_net("MIPI_D0_N", NetClassKind::Signal);
    let d1p = clean.add_net("MIPI_D1_P", NetClassKind::Signal);
    let d1n = clean.add_net("MIPI_D1_N", NetClassKind::Signal);
    let auxp = clean.add_net("HDMI_AUX_P", NetClassKind::Signal);
    let auxn = clean.add_net("HDMI_AUX_N", NetClassKind::Signal);
    for (net, y) in [
        (d0p, 4.0),
        (d0n, 5.0),
        (d1p, 7.0),
        (d1n, 8.0),
        (auxp, 11.0),
        (auxn, 12.0),
    ] {
        clean.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(8.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let clean_report = audit(&clean, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean_report.diff_pair_layer_mismatch_violations, 0,
        "each pair is internally routed on one layer set"
    );
    assert_eq!(
        clean_report.diff_pair_interface_layer_mismatch_violations, 0,
        "same-interface MIPI_D0/D1 pairs share the same layer set"
    );
    assert_eq!(
        clean_report.diff_pair_interface_via_count_mismatch_violations, 0,
        "same-interface MIPI_D0/D1 pairs use the same total via count"
    );

    let mut dirty = clean.clone();
    for track in dirty
        .tracks
        .iter_mut()
        .filter(|track| track.net == d1p || track.net == d1n)
    {
        track.layer = LayerId(1);
    }
    let dirty_report = audit(&dirty, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty_report.diff_pair_layer_mismatch_violations, 0,
        "each individual pair still has matched P/N layer sets"
    );
    assert_eq!(
        dirty_report.diff_pair_interface_layer_mismatch_violations, 1,
        "MIPI_D1 routed on a different layer set than MIPI_D0 violates same-interface routing"
    );
    assert_eq!(
        dirty_report.diff_pair_interface_via_count_mismatch_violations, 0,
        "the layer-mismatch fixture keeps total via counts matched"
    );
    assert!(
        !dirty_report.hard_drc_clean(),
        "same-interface differential-pair layer mismatch rejects clean-board selection"
    );

    let mut via_mismatched = clean;
    for (net, x, y) in [(d1p, 4.0, 7.0), (d1n, 4.0, 8.0)] {
        via_mismatched.vias.push(Via {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
    }
    let via_report = audit(&via_mismatched, &[], &[], &DesignRules::holohv());
    assert_eq!(
        via_report.diff_pair_layer_mismatch_violations, 0,
        "each individual pair still has matched P/N layer sets"
    );
    assert_eq!(
        via_report.diff_pair_via_count_violations, 0,
        "D1 P/N members use the same via count, so the per-pair via rule stays clean"
    );
    assert_eq!(
        via_report.diff_pair_interface_layer_mismatch_violations, 0,
        "all interface pairs still route on the same layer set"
    );
    assert_eq!(
        via_report.diff_pair_interface_via_count_mismatch_violations, 1,
        "MIPI_D1 using more total vias than MIPI_D0 violates same-interface matching"
    );
    assert!(
        !via_report.hard_drc_clean(),
        "same-interface differential-pair via-count mismatch rejects clean-board selection"
    );
}

#[test]
fn detects_differential_pair_via_station_mismatch() {
    use crate::board::{Via, ViaKind};

    let mut b = board();
    let p = b.add_net("MIPI_P", NetClassKind::Signal);
    let n = b.add_net("MIPI_N", NetClassKind::Signal);
    for (net, y) in [(p, 4.0), (n, 5.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    for (net, x, y) in [(p, 6.0, 4.0), (n, 6.3, 5.0)] {
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
    }
    let matched = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        matched.diff_pair_via_count_violations, 0,
        "both pair members use one via"
    );
    assert_eq!(
        matched.diff_pair_via_symmetry_violations, 0,
        "0.3 mm via station mismatch stays inside the 0.5 mm tolerance"
    );

    b.vias[1].pos.x = Nm::from_mm(8.0);
    let shifted = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        shifted.diff_pair_via_count_violations, 0,
        "equal via count alone is not enough to prove via symmetry"
    );
    assert_eq!(
        shifted.diff_pair_via_symmetry_violations, 1,
        "P/N vias at different routing stations are flagged"
    );
}

#[test]
fn detects_differential_pair_length_mismatch() {
    let mut mismatched = board();
    let p = mismatched.add_net("DATA_P", NetClassKind::Signal);
    let n = mismatched.add_net("DATA_N", NetClassKind::Signal);
    mismatched.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: p,
    });
    mismatched.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: n,
    });
    let dirty = audit(&mismatched, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.diff_pair_length_mismatch_violations, 1,
        "pair members whose routed lengths differ by more than tolerance are flagged"
    );

    let mut matched = board();
    let p2 = matched.add_net("ADDR_P", NetClassKind::Signal);
    let n2 = matched.add_net("ADDR_N", NetClassKind::Signal);
    for (net, y, end_x) in [(p2, 4.0, 8.0), (n2, 5.0, 8.3)] {
        matched.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(end_x), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let clean = audit(&matched, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_length_mismatch_violations, 0,
        "0.3 mm intra-pair mismatch stays inside the 0.5 mm tolerance"
    );
}

#[test]
fn detects_differential_pair_segment_length_mismatch() {
    use crate::board::{Via, ViaKind};

    let mut b = board();
    let p = b.add_net("LANE_P", NetClassKind::Signal);
    let n = b.add_net("LANE_N", NetClassKind::Signal);
    for (net, y, layer0_end_x, layer1_start_x) in [(p, 4.0, 8.0, 8.0), (n, 5.0, 4.0, 4.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(layer0_end_x), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(layer1_start_x), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net,
        });
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(layer0_end_x), Nm::from_mm(y)),
            drill: Nm::from_mm(0.2),
            diameter: Nm::from_mm(0.46),
            net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Micro,
            filled: false,
        });
    }

    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_length_mismatch_violations, 0,
        "total routed length remains matched"
    );
    assert_eq!(
        r.diff_pair_via_count_violations, 0,
        "both pair members use the same via count"
    );
    assert_eq!(
        r.diff_pair_segment_length_mismatch_violations, 1,
        "per-layer differential-pair segment skew is flagged even when total length matches"
    );
}

#[test]
fn detects_parallel_bus_length_mismatch() {
    let mut b = board();
    let d0 = b.add_net("BUS_D0", NetClassKind::Signal);
    let d1 = b.add_net("BUS_D1", NetClassKind::Signal);
    let tx0 = b.add_net("TX_0", NetClassKind::Signal);
    let tx1 = b.add_net("TX_1", NetClassKind::Signal);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(9.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: d0,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(4.0)),
        end: Point::new(Nm::from_mm(11.0), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: d1,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx0,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx1,
    });

    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.parallel_bus_length_mismatch_violations, 0,
        "BUS_D0/BUS_D1 differ by exactly the 2 mm bus-skew budget, and TX_0/TX_1 are not bus-grouped"
    );

    b.tracks[1].end = Point::new(Nm::from_mm(11.5), Nm::from_mm(4.0));
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.parallel_bus_length_mismatch_violations, 1,
        "BUS_D0/BUS_D1 differ by 2.5 mm, exceeding the configured parallel-bus skew budget"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "parallel bus skew must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_differential_pair_pad_entry_mismatch() {
    let mut b = board();
    let p = b.add_net("PAD_ENTRY_P", NetClassKind::Signal);
    let n = b.add_net("PAD_ENTRY_N", NetClassKind::Signal);
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
        layers: vec![LayerId(0)],
        net: Some(p),
    });
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(n),
    });

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.2), Nm::from_mm(4.0)),
        end: Point::new(Nm::from_mm(10.0), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: p,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.4), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(10.2), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: n,
    });
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_pad_entry_mismatch_violations, 0,
        "0.2 mm P/N pad-entry mismatch stays inside the 0.5 mm local budget"
    );
    assert_eq!(
        clean.diff_pair_length_mismatch_violations, 0,
        "the fixture keeps total routed P/N length matched"
    );

    b.tracks[1].start = Point::new(Nm::from_mm(3.0), Nm::from_mm(5.0));
    b.tracks[1].end = Point::new(Nm::from_mm(10.8), Nm::from_mm(5.0));
    let mismatched = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        mismatched.diff_pair_pad_entry_mismatch_violations, 1,
        "1.0 mm versus 0.2 mm pad-entry breakout is flagged independently of total length"
    );
    assert_eq!(
        mismatched.diff_pair_length_mismatch_violations, 0,
        "the over-budget pad-entry mismatch is not a total route-length mismatch"
    );
}

#[test]
fn detects_differential_pair_pad_entry_length() {
    let mut b = board();
    let p = b.add_net("LONG_ENTRY_P", NetClassKind::Signal);
    let n = b.add_net("LONG_ENTRY_N", NetClassKind::Signal);
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(4.0)),
        layers: vec![LayerId(0)],
        net: Some(p),
    });
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(n),
    });
    for (net, y) in [(p, 4.0), (n, 5.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(3.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_pad_entry_mismatch_violations, 0,
        "equal 1 mm pad entries are symmetric"
    );
    assert_eq!(
        clean.diff_pair_pad_entry_length_violations, 0,
        "1 mm pad entries stay inside the 2 mm local breakout budget"
    );

    for track in &mut b.tracks {
        track.start.x = Nm::from_mm(5.0);
        track.end.x = Nm::from_mm(12.0);
    }
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.diff_pair_pad_entry_mismatch_violations, 0,
        "equal 3 mm pad entries are still symmetric"
    );
    assert_eq!(
        dirty.diff_pair_pad_entry_length_violations, 1,
        "matched-but-long 3 mm pad entries violate the 2 mm local breakout budget once per pair"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "overlength differential-pair pad entries must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_differential_pair_spacing_variation() {
    let mut clean = board();
    let p = clean.add_net("USB_P", NetClassKind::Signal);
    let n = clean.add_net("USB_N", NetClassKind::Signal);
    for (net, y0, y1) in [(p, 4.0, 4.0), (n, 4.6, 4.8)] {
        clean.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y0)),
            end: Point::new(Nm::from_mm(6.0), Nm::from_mm(y0)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
        clean.tracks.push(Track {
            start: Point::new(Nm::from_mm(8.0), Nm::from_mm(y1)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y1)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let within = audit(&clean, &[], &[], &DesignRules::holohv());
    assert_eq!(
        within.diff_pair_spacing_variation_violations, 0,
        "0.2 mm pair-spacing variation stays inside the 0.25 mm tolerance"
    );

    let mut dirty = clean.clone();
    dirty.tracks[3].start.y = Nm::from_mm(5.2);
    dirty.tracks[3].end.y = Nm::from_mm(5.2);
    let widened = audit(&dirty, &[], &[], &DesignRules::holohv());
    assert_eq!(
        widened.diff_pair_spacing_variation_violations, 1,
        "pair spacing that opens by more than tolerance is flagged"
    );
}

#[test]
fn detects_differential_pair_keepout() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
    use crate::place::rotation::{Rot};

    let track = |b: &mut Board, net, y: f64| {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(10.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    };

    let mut signal = board();
    let p = signal.add_net("DATA_P", NetClassKind::Signal);
    let n = signal.add_net("DATA_N", NetClassKind::Signal);
    let other = signal.add_net("OTHER", NetClassKind::Signal);
    track(&mut signal, p, 4.0);
    track(&mut signal, n, 4.6);
    track(&mut signal, other, 5.35);
    let r = audit(&signal, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_keepout_violations, 1,
        "an unrelated signal inside the 30 mil differential-pair keepout is flagged once"
    );

    let mut clock = board();
    let p = clock.add_net("CLK_P", NetClassKind::Signal);
    let n = clock.add_net("CLK_N", NetClassKind::Signal);
    let other = clock.add_net("OTHER", NetClassKind::Signal);
    track(&mut clock, p, 4.0);
    track(&mut clock, n, 4.6);
    track(&mut clock, other, 5.55);
    let r = audit(&clock, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_keepout_violations, 1,
        "clock pairs use the wider 50 mil keepout"
    );

    let mut pair_to_pair = board();
    let ap = pair_to_pair.add_net("A_P", NetClassKind::Signal);
    let an = pair_to_pair.add_net("A_N", NetClassKind::Signal);
    let bp = pair_to_pair.add_net("B_P", NetClassKind::Signal);
    let bn = pair_to_pair.add_net("B_N", NetClassKind::Signal);
    track(&mut pair_to_pair, ap, 4.0);
    track(&mut pair_to_pair, an, 4.6);
    track(&mut pair_to_pair, bp, 5.2);
    track(&mut pair_to_pair, bn, 5.8);
    let r = audit(&pair_to_pair, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.diff_pair_keepout_violations, 1,
        "adjacent differential pairs closer than 5W are flagged once per pair relationship"
    );

    let mut component_blocked = board();
    let p = component_blocked.add_net("USB_P", NetClassKind::Signal);
    let n = component_blocked.add_net("USB_N", NetClassKind::Signal);
    let quiet = component_blocked.add_net("GPIO", NetClassKind::Signal);
    track(&mut component_blocked, p, 4.0);
    track(&mut component_blocked, n, 4.6);
    let lib = vec![FootprintDef::new(
        "R_0402",
        (Nm::from_mm(0.8), Nm::from_mm(0.8)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let blocker = Component {
        fp: 0,
        nets: vec![Some(quiet)],
        refdes: "R1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(4.3)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let blocked = audit(&component_blocked, &[blocker], &lib, &DesignRules::holohv());
    assert_eq!(
        blocked.diff_pair_violations, 1,
        "an unrelated component courtyard between P/N members creates a differential-pair obstruction"
    );
    assert!(
        !blocked.hard_drc_clean(),
        "component intrusion between differential-pair members rejects optimizer clean-board selection"
    );
}

#[test]
fn detects_asymmetric_diff_pair_coupling_caps() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

    let mut matched = board();
    let p = matched.add_net("MGT_P", NetClassKind::Signal);
    let n = matched.add_net("MGT_N", NetClassKind::Signal);

    for (net, y) in [(p, 4.0), (n, 5.0)] {
        matched.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y)),
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
    let lib = vec![cap_fp];
    let cap = |net, refdes: &str, x, y| Component {
        fp: 0,
        nets: vec![Some(net), None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let matched_comps = vec![cap(p, "C_AC_P", 6.0, 4.0), cap(n, "C_AC_N", 6.3, 5.0)];
    let clean = audit(&matched, &matched_comps, &lib, &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_coupling_cap_symmetry_violations, 0,
        "P/N coupling caps whose pair-axis stations differ by 0.3 mm stay inside tolerance"
    );

    let shifted_comps = vec![cap(p, "C_AC_P", 6.0, 4.0), cap(n, "C_AC_N", 8.0, 5.0)];
    let shifted = audit(&matched, &shifted_comps, &lib, &DesignRules::holohv());
    assert_eq!(
        shifted.diff_pair_coupling_cap_symmetry_violations, 1,
        "P/N coupling caps shifted by more than 0.5 mm along the pair axis are flagged"
    );

    let one_sided = audit(
        &matched,
        &[cap(p, "C_AC_P", 6.0, 4.0)],
        &lib,
        &DesignRules::holohv(),
    );
    assert_eq!(
        one_sided.diff_pair_coupling_cap_symmetry_violations, 1,
        "a coupling capacitor on only one leg is not symmetric"
    );
}

#[test]
fn detects_oversized_diff_pair_coupling_cap_packages() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
    use crate::place::rotation::{Rot};

    let mut b = board();
    let p = b.add_net("USB_P", NetClassKind::Signal);
    let n = b.add_net("USB_N", NetClassKind::Signal);

    for (net, y) in [(p, 4.0), (n, 5.0)] {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    }
    let cap_fp = |name: &str, w_mm: f64, h_mm: f64| {
        FootprintDef::new(
            name,
            (Nm::from_mm(w_mm), Nm::from_mm(h_mm)),
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
        )
    };
    let cap = |fp, net, refdes: &str, x, y| Component {
        fp,
        nets: vec![Some(net), None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    let clean_lib = vec![cap_fp("C0603", 1.6, 0.8)];
    let clean_comps = vec![cap(0, p, "C_AC_P", 6.0, 4.0), cap(0, n, "C_AC_N", 6.0, 5.0)];
    let clean = audit(&b, &clean_comps, &clean_lib, &DesignRules::holohv());
    assert_eq!(
        clean.diff_pair_coupling_cap_symmetry_violations, 0,
        "the symmetric fixture isolates package size from placement symmetry"
    );
    assert_eq!(
        clean.diff_pair_coupling_cap_package_violations, 0,
        "0603-class coupling capacitors stay inside the 1.7 mm package budget"
    );

    let dirty_lib = vec![cap_fp("C0805", 2.0, 1.25)];
    let dirty_comps = vec![cap(0, p, "C_AC_P", 6.0, 4.0), cap(0, n, "C_AC_N", 6.0, 5.0)];
    let dirty = audit(&b, &dirty_comps, &dirty_lib, &DesignRules::holohv());
    assert_eq!(
        dirty.diff_pair_coupling_cap_symmetry_violations, 0,
        "oversized but symmetric coupling capacitors are not a symmetry violation"
    );
    assert_eq!(
        dirty.diff_pair_coupling_cap_package_violations, 2,
        "both 0805-class coupling capacitors exceed the 0603-class courtyard budget"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "oversized differential-pair coupling capacitors must reject optimizer clean-board selection"
    );
}
