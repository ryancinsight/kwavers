use super::*;

#[test]
fn clean_board_has_no_faults() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    let p0 = Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0));
    let p1 = Point::new(Nm::from_mm(8.0), Nm::from_mm(2.0));
    b.add_pad(Pad {
        pos: p0,
        layers: vec![LayerId(0)],
        net: Some(n),
    });
    b.add_pad(Pad {
        pos: p1,
        layers: vec![LayerId(0)],
        net: Some(n),
    });
    b.tracks.push(Track {
        start: p0,
        end: p1,
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: n,
    });
    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        (r.crossings, r.near_shorts, r.dangling),
        (0, 0, 0),
        "single connected net is clean"
    );
}

#[test]
fn detects_via_in_surge_suppressor_connector_path() {
    use crate::place::footprint::{PadDef, Role};
    use crate::place::rotation::Rot;
    use crate::place::Placement;

    let lib = vec![
        FootprintDef::new(
            "J",
            (Nm::from_mm(3.0), Nm::from_mm(3.0)),
            Role::Connector,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "TVS",
            (Nm::from_mm(1.0), Nm::from_mm(0.6)),
            Role::Passive,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let mut clean = board();
    let incoming = clean.add_net("USB_IN", NetClassKind::Signal);
    let comps = vec![
        Component {
            fp: 0,
            nets: vec![Some(incoming)],
            refdes: "J1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        },
        Component {
            fp: 1,
            nets: vec![Some(incoming)],
            refdes: "TVS1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        },
    ];
    clean.vias.push(Via {
        pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.45),
        net: incoming,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });
    let clean_report = audit(&clean, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        clean_report.surge_suppressor_via_violations, 0,
        "a same-net via beyond the suppressor is not in the connector-to-clamp path"
    );

    let mut dirty = clean;
    dirty.vias[0].pos = Point::new(Nm::from_mm(5.0), Nm::from_mm(10.0));
    let dirty_report = audit(&dirty, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        dirty_report.surge_suppressor_via_violations, 1,
        "a via in the incoming connector-to-suppressor segment adds parasitic inductance"
    );
    assert!(
        !dirty_report.hard_drc_clean(),
        "connector-to-suppressor vias must reject clean-board selection"
    );
}

#[test]
fn audit_detects_isolation_and_ac_coupling() {
    use crate::board::{LayerId, Pad, Track};
    use crate::place::footprint::Role;
    use crate::place::rotation::Rot;
    use crate::place::Placement;

    let mut b = board();
    // Setup net classes
    let hv = b.add_net("TRIG_HV", NetClassKind::Hv);
    let lv = b.add_net("CTRL_LV", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    // Add pads for the nets to give them coordinates
    let p_hv = Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0));
    let p_lv = Point::new(Nm::from_mm(10.0), Nm::from_mm(5.0));
    b.add_pad(Pad {
        pos: p_hv,
        layers: vec![LayerId(0)],
        net: Some(hv),
    });
    b.add_pad(Pad {
        pos: p_lv,
        layers: vec![LayerId(0)],
        net: Some(lv),
    });

    // Add components that bridge the isolation boundary
    let lib = vec![FootprintDef {
        name: "R_0603".to_string(),
        pads: vec![],
        courtyard: (Nm::from_mm(1.6), Nm::from_mm(0.8)),
        role: Role::Passive,
        rotation_policy: crate::place::rotation::RotationPolicy::HalfTurn,
        pad_names: vec![],
        model: None,
        ball_pitch: None,
        i_dd_a: 0.0,
        capacitance_f: 0.0,
        dielectric_grade: crate::place::footprint::DielectricGrade::Unknown,
        package_form_factor: crate::place::footprint::PackageFormFactor::Unknown,
    }];

    // Low-voltage control net source component (connector J2)
    let j2 = Component {
        fp: 0,
        nets: vec![Some(lv)],
        refdes: "J2".to_string(),
        placement: Placement {
            pos: p_lv,
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    // Component U1 (the HV pulser) connected to TRIG_HV
    let u1 = Component {
        fp: 0,
        nets: vec![Some(hv)],
        refdes: "U1".to_string(),
        placement: Placement {
            pos: p_hv,
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    // Resistor directly bridging them (R1)
    let r1 = Component {
        fp: 0,
        nets: vec![Some(hv), Some(lv)],
        refdes: "R1".to_string(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(7.5), Nm::from_mm(5.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    let comps = vec![j2, u1, r1];

    // Parallel switching tracks causing AC coupling
    // Track 1: switching net
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(1.0)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: hv,
    });
    // Track 2: GND net, adjacent coplanar spacing = 0.3mm (within 2mm spacing limit)
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.5)),
        end: Point::new(Nm::from_mm(3001.0), Nm::from_mm(1.5)),
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: gnd,
    });

    let r = audit(&b, &comps, &lib, &DesignRules::holohv());

    assert_eq!(
        r.isolation_violations, 1,
        "direct LV to HV bridge should violate isolation"
    );
    assert_eq!(
        r.ac_coupling_violations, 1,
        "close parallel HV to GND track should violate AC coupling"
    );
    assert!(
        !r.hotspots.is_empty(),
        "hotspots must contain violation coordinates"
    );
    assert!(
        r.risk_score > 60.0,
        "risk score must include isolation and AC coupling penalties"
    );
}
