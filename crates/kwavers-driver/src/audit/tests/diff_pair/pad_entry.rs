use super::super::*;

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
