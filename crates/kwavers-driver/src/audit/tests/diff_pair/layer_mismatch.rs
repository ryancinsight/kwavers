use super::super::*;

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
