use super::*;

#[test]
fn point_feature_clearance_is_layer_aware() {
    // A 0→1 HDI micro-via of net A and a foreign-net track passing 0.1 mm from its centre — well
    // inside the via's 0.125 mm copper plus clearance. On layer 3 (which the via barrel does NOT
    // reach) there is no copper to clash with, so per-layer DRC reports nothing; the same track on
    // layer 0 (which the via spans) is a real clearance violation. This guards the layer gate that
    // stops a buried/micro via from being false-flagged against tracks on layers it never touches.
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
    let a = b.add_net("A", NetClassKind::Signal);
    let bn = b.add_net("B", NetClassKind::Signal);
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
        net: a,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        drill: Nm::from_mm(0.1),
        diameter: Nm::from_mm(0.25),
        filled: true,
    });
    let track = |layer| Track {
        start: Point::new(Nm::from_mm(10.1), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(10.1), Nm::from_mm(12.0)),
        width: Nm::from_mm(0.15),
        layer,
        net: bn,
    };

    b.tracks.push(track(LayerId(3)));
    assert_eq!(
        audit(&b, &[], &[], &DesignRules::holohv()).clearance_violations,
        0,
        "a 0→1 micro-via has no copper on layer 3, so a foreign track there is not a clearance violation"
    );

    b.tracks[0] = track(LayerId(0));
    assert_eq!(
        audit(&b, &[], &[], &DesignRules::holohv()).clearance_violations,
        1,
        "the same track on layer 0 — which the via barrel spans — is a real clearance violation"
    );
}

#[test]
fn serpentine_spacing_uses_edge_gap_not_centerline_gap() {
    let mut b = board();
    let sig = b.add_net("SERP", NetClassKind::Signal);
    let make_segment = |y_mm| Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(y_mm)),
        end: Point::new(Nm::from_mm(12.0), Nm::from_mm(y_mm)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: sig,
    };

    b.tracks.push(make_segment(4.0));
    b.tracks.push(make_segment(4.675));
    let edge_violation = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        edge_violation.serpentine_spacing_violations, 1,
        "0.675 mm centerline spacing is 4.5W, but the 0.525 mm copper edge gap is below 4W"
    );

    b.tracks[1] = make_segment(4.75);
    let edge_clear = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        edge_clear.serpentine_spacing_violations, 0,
        "0.75 mm centerline spacing gives the required 0.6 mm copper edge gap for 0.15 mm traces"
    );
}

#[test]
fn serpentine_compensation_must_stay_near_bend_root() {
    let spec = GridSpec::cover(Nm::from_mm(70.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    let mut remote = Board::new(spec);
    let net = remote.add_net("SERP_REMOTE", NetClassKind::Signal);
    let h = |x0, x1, y| Track {
        start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
        end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    };
    let v = |x, y0, y1| Track {
        start: Point::new(Nm::from_mm(x), Nm::from_mm(y0)),
        end: Point::new(Nm::from_mm(x), Nm::from_mm(y1)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    };
    remote.tracks.push(h(2.0, 50.0, 10.0));
    remote.tracks.push(v(2.0, 10.0, 10.8));
    remote.tracks.push(h(2.0, 50.0, 10.8));
    remote.tracks.push(v(50.0, 10.0, 10.8));

    let remote_report = audit(&remote, &[], &[], &DesignRules::holohv());
    assert_eq!(
        remote_report.serpentine_spacing_violations, 0,
        "0.65 mm edge gap is wider than the 4W spacing budget for 0.15 mm traces"
    );
    assert_eq!(
        remote_report.serpentine_compensation_distance_violations, 1,
        "the parallel compensation midpoint is 24 mm from the nearest bend, above the 15 mm guide budget"
    );

    let mut local = Board::new(spec);
    let net = local.add_net("SERP_LOCAL", NetClassKind::Signal);
    let h = |x0, x1, y| Track {
        start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
        end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    };
    let v = |x, y0, y1| Track {
        start: Point::new(Nm::from_mm(x), Nm::from_mm(y0)),
        end: Point::new(Nm::from_mm(x), Nm::from_mm(y1)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    };
    local.tracks.push(h(2.0, 20.0, 10.0));
    local.tracks.push(v(2.0, 10.0, 10.8));
    local.tracks.push(h(2.0, 20.0, 10.8));
    local.tracks.push(v(20.0, 10.0, 10.8));

    let local_report = audit(&local, &[], &[], &DesignRules::holohv());
    assert_eq!(
        local_report.serpentine_compensation_distance_violations, 0,
        "the local compensation midpoint is 9 mm from a bend, inside the 15 mm guide budget"
    );
}

#[test]
fn sharp_bend_detection_rejects_acute_bends_and_accepts_one_thirty_five() {
    let mut acute = board();
    let net = acute.add_net("TX_ACUTE", NetClassKind::Signal);
    acute.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    acute.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(2.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    let acute_report = audit(&acute, &[], &[], &DesignRules::holohv());
    assert_eq!(
        acute_report.sharp_bends, 1,
        "a 45 degree same-net bend is sharper than the guide's 135 degree routing geometry"
    );

    let mut obtuse = board();
    let net = obtuse.add_net("TX_OBTUSE", NetClassKind::Signal);
    obtuse.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    obtuse.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net,
    });
    let obtuse_report = audit(&obtuse, &[], &[], &DesignRules::holohv());
    assert_eq!(
        obtuse_report.sharp_bends, 0,
        "a 135 degree same-net bend follows the guide's preferred bend geometry"
    );
}

#[test]
fn track_crossing_detection_rejects_opposed_diagonals_inside_one_cell() {
    let mut dirty = board();
    let a = dirty.add_net("A", NetClassKind::Signal);
    let b = dirty.add_net("B", NetClassKind::Signal);
    dirty.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(3.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: a,
    });
    dirty.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(3.0)),
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: b,
    });

    let dirty_report = audit(&dirty, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty_report.track_crossing_violations, 1,
        "different-net diagonal tracks crossing inside a grid cell are a KiCad DRC failure"
    );
    assert!(
        !dirty_report.hard_drc_clean(),
        "track crossings must reject optimizer clean-board selection"
    );

    let mut clean = board();
    let a = clean.add_net("A", NetClassKind::Signal);
    let b = clean.add_net("B", NetClassKind::Signal);
    clean.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(3.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: a,
    });
    clean.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(3.0)),
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: b,
    });

    let clean_report = audit(&clean, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean_report.track_crossing_violations, 0,
        "crossing geometry on different copper layers is not a same-layer DRC crossing"
    );
}

#[test]
fn detects_crossing_diagonal_tracks_of_different_nets() {
    // The dominant real-board failure: two 45° tracks of different nets crossing on one layer.
    // Their centre-lines intersect ⇒ edge-gap is negative ⇒ a hard clearance violation that
    // kicad-cli flags as `tracks_crossing` + `clearance`. The audit must see it too, else the
    // cooptimize judge is blind to it and never optimises it out.
    let mut b = board();
    let a = b.add_net("A", NetClassKind::Signal);
    let c = b.add_net("C", NetClassKind::Signal);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(6.0), Nm::from_mm(6.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: a,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(6.0)),
        end: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: c,
    });
    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert!(
        r.clearance_violations >= 1,
        "crossing diagonal tracks of different nets must register as a clearance violation, got {}",
        r.clearance_violations
    );
}

#[test]
fn detects_via_hole_too_close_to_foreign_track() {
    // A via whose drilled barrel passes within the hole-to-copper clearance of a foreign-net
    // track on a layer the barrel spans — kicad-cli's `hole_clearance` class, previously
    // unmodelled by the audit so escape-via boards passed internally but failed externally.
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    let f = b.add_net("F", NetClassKind::Signal);
    let rules = DesignRules::holohv();
    // Via barrel F.Cu..In1 at (5,5), drill 0.3 mm (radius 0.15).
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.5),
        net: n,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Blind,
        filled: false,
    });
    // Foreign track on In1 (layer 1, in the barrel span) passing 0.2 mm from the via centre:
    // hole-edge gap = 0.2 − 0.15(drill) − 0.075(half-width) = −0.025 mm < clearance ⇒ violation.
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.2)),
        end: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.2)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: f,
    });
    let r = audit(&b, &[], &[], &rules);
    assert_eq!(
        r.hole_clearance_violations, 1,
        "via hole within clearance of a foreign track on a spanned layer is a violation"
    );
    // A same-net track at the same spot is the via's own connection — never a hole violation.
    let mut b2 = board();
    let n2 = b2.add_net("N", NetClassKind::Signal);
    b2.vias.push(Via {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.5),
        net: n2,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Blind,
        filled: false,
    });
    b2.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(5.2)),
        end: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.2)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: n2,
    });
    assert_eq!(
        audit(&b2, &[], &[], &rules).hole_clearance_violations,
        0,
        "same-net copper at the via is its own connection, not a hole violation"
    );
}

#[test]
fn detects_dangling_track_end() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    // A track whose start is on a pad but whose end floats in space.
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        layers: vec![LayerId(0)],
        net: Some(n),
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: n,
    });
    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(r.dangling, 1, "the floating track end is an antenna");
}

#[test]
fn detects_unsplit_track_body_junction_as_dangling() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    for pos in [
        Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
        Point::new(Nm::from_mm(3.0), Nm::from_mm(4.0)),
    ] {
        b.add_pad(Pad {
            pos,
            layers: vec![LayerId(0)],
            net: Some(n),
        });
    }
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: n,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(3.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: n,
    });

    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.dangling, 1,
        "an endpoint on an unsplit same-net track body still emits a KiCad track_dangling warning"
    );
}

#[test]
fn detects_high_speed_stub_branch() {
    let mut branched = board();
    let tx = branched.add_net("TX_STUB", NetClassKind::Signal);
    let node = Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0));
    for (start, end) in [
        (Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)), node),
        (node, Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0))),
        (node, Point::new(Nm::from_mm(8.0), Nm::from_mm(12.0))),
    ] {
        branched.tracks.push(Track {
            start,
            end,
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net: tx,
        });
    }
    let r = audit(&branched, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.high_speed_stub_violations, 1,
        "one TX T-junction is a connected high-speed stub"
    );

    let mut daisy = board();
    let tx2 = daisy.add_net("TX_DAISY", NetClassKind::Signal);
    daisy.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx2,
    });
    daisy.tracks.push(Track {
        start: Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0)),
        end: Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx2,
    });
    let clean = audit(&daisy, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.high_speed_stub_violations, 0,
        "a two-segment daisy-chain has no branch node"
    );
}
