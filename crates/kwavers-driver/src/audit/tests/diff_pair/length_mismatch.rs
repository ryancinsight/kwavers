use super::super::*;

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
