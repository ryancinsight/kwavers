use super::*;

#[test]
fn detects_oversized_high_speed_via_diameter() {
    use crate::board::{Via, ViaKind};

    let mut b = board();
    let tx = b.add_net("TX_VIA_SIZE", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.pads.push(Pad {
        pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        layers: vec![LayerId(0)],
        net: Some(tx),
    });
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(5.5), Nm::from_mm(5.0)),
        drill: Nm::from_mm(0.2),
        diameter: DesignRules::holohv().via_diameter(),
        net: tx,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(5.5), Nm::from_mm(5.5)),
        drill: Nm::from_mm(0.2),
        diameter: DesignRules::holohv().via_diameter(),
        net: gnd,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });

    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.high_speed_via_pad_proximity_violations, 0,
        "the high-speed via is within the same-net pad proximity budget"
    );
    assert_eq!(
        clean.high_speed_transition_ground_via_violations, 0,
        "the high-speed via has a local ground transition via"
    );
    assert_eq!(
        clean.high_speed_via_diameter_violations, 0,
        "the default rule-sized high-speed via is accepted"
    );

    b.vias[0].diameter = Nm::from_mm(0.7);
    let oversized = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        oversized.high_speed_via_pad_proximity_violations, 0,
        "oversizing the via does not change its same-net pad distance"
    );
    assert_eq!(
        oversized.high_speed_transition_ground_via_violations, 0,
        "oversizing the via does not remove the local ground transition via"
    );
    assert_eq!(
        oversized.high_speed_via_diameter_violations, 1,
        "a high-speed via larger than the selected rule diameter is flagged"
    );
}

#[test]
fn detects_oversized_blind_and_buried_via_drills() {
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
    let sig = b.add_net("SIG_BLIND_BURIED_DRILL", NetClassKind::Signal);

    let via = |kind: ViaKind, drill_mm: f64, x_mm: f64| Via {
        pos: Point::new(Nm::from_mm(x_mm), Nm::from_mm(5.0)),
        drill: Nm::from_mm(drill_mm),
        diameter: Nm::from_mm(drill_mm + 0.15),
        net: sig,
        from: match kind {
            ViaKind::Buried => LayerId(1),
            _ => LayerId(0),
        },
        to: match kind {
            ViaKind::Buried => LayerId(2),
            ViaKind::Blind => LayerId(2),
            ViaKind::Micro | ViaKind::Through => LayerId(3),
        },
        kind,
        filled: false,
    };

    b.vias.push(via(ViaKind::Blind, 0.15, 4.0));
    b.vias.push(via(ViaKind::Buried, 0.15, 6.0));
    b.vias.push(via(ViaKind::Through, 0.30, 8.0));
    let clean = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.blind_buried_via_drill_violations, 0,
        "rule-sized blind/buried vias and larger through vias are accepted"
    );

    b.vias.push(via(ViaKind::Blind, 0.16, 10.0));
    b.vias.push(via(ViaKind::Buried, 0.20, 12.0));
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.blind_buried_via_drill_violations, 2,
        "only oversized blind and buried via drills violate the 0.15 mm fabrication limit"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "oversized blind/buried vias must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_high_speed_via_stub() {
    use crate::board::{Via, ViaKind};

    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut dirty = Board::new(spec);
    let tx = dirty.add_net("TX_STUBBED_VIA", NetClassKind::Signal);
    let p = Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0));
    dirty.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(8.0)),
        end: p,
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx,
    });
    dirty.tracks.push(Track {
        start: p,
        end: Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: tx,
    });
    dirty.vias.push(Via {
        pos: p,
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: tx,
        from: LayerId(0),
        to: LayerId(3),
        kind: ViaKind::Through,
        filled: false,
    });
    let r = audit(&dirty, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.high_speed_via_stub_violations, 1,
        "a full-stack via used only on layers 0..1 leaves a high-speed via stub"
    );

    let mut clean = dirty.clone();
    clean.vias[0].to = LayerId(1);
    clean.vias[0].kind = ViaKind::Micro;
    let r = audit(&clean, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.high_speed_via_stub_violations, 0,
        "a via whose physical span matches the used signal layers has no stub"
    );
}
