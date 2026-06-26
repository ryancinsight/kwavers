//! Decoupling cap placement tests: ground via proximity, power layer matching,
//! commutation loop area, and active-IC internal power plane.

use super::*;
use crate::place::component::Placement;
use crate::place::footprint::PadDef;

#[test]
fn detects_decoupling_cap_without_local_ground_via() {
    let mut b = board();
    let pwr = b.add_net("VDD", NetClassKind::Power);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let lib = vec![FootprintDef::new(
        "C0402",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![
            crate::place::PadDef {
                offset: Point::new(Nm::from_mm(-0.3), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            crate::place::PadDef {
                offset: Point::new(Nm::from_mm(0.3), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let comps = vec![Component {
        fp: 0,
        nets: vec![Some(pwr), Some(gnd)],
        refdes: "C1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: crate::place::Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];

    let dirty = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        dirty.decoupling_ground_via_violations, 1,
        "an SMD decoupling ground pad without a local ground via is flagged"
    );

    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(14.0), Nm::from_mm(10.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: gnd,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });
    let far = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        far.decoupling_ground_via_violations, 1,
        "a ground via outside the 1 mm decoupling budget is not local"
    );

    b.vias[0].pos = Point::new(Nm::from_mm(10.5), Nm::from_mm(10.0));
    let clean = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        clean.decoupling_ground_via_violations, 0,
        "a nearby ground via clears the decoupling local-return requirement"
    );
}

#[test]
fn detects_decoupling_power_pin_on_opposite_layer() {
    let mut b = board();
    let pwr = b.add_net("VDD", NetClassKind::Power);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.vias.push(Via {
        pos: Point::new(Nm::from_mm(6.2), Nm::from_mm(5.0)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.45),
        net: gnd,
        from: LayerId(0),
        to: LayerId(1),
        kind: ViaKind::Micro,
        filled: false,
    });
    let cap_fp = FootprintDef::new(
        "C0402",
        (Nm::from_mm(1.0), Nm::from_mm(0.5)),
        Role::Decoupling,
        vec![
            crate::place::PadDef {
                offset: Point::new(Nm::from_mm(-0.2), Nm(0)),
                size: (Nm::from_mm(0.2), Nm::from_mm(0.2)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            crate::place::PadDef {
                offset: Point::new(Nm::from_mm(0.2), Nm(0)),
                size: (Nm::from_mm(0.2), Nm::from_mm(0.2)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let ic_fp = |power_layer| {
        FootprintDef::new(
            "U",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![crate::place::PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![power_layer],
                power_pin: true,
            }],
        )
    };
    let comps = vec![
        Component {
            fp: 1,
            nets: vec![Some(pwr)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
                rot: crate::place::Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        },
        Component {
            fp: 0,
            nets: vec![Some(pwr), Some(gnd)],
            refdes: "C1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(5.0)),
                rot: crate::place::Rot::R0,
            },
            assoc_ic: Some(0),
            locked: false,
            ..Default::default()
        },
    ];

    let clean_lib = vec![cap_fp.clone(), ic_fp(LayerId(0))];
    let clean = audit(&b, &comps, &clean_lib, &DesignRules::holohv());
    assert_eq!(
        clean.decoupling_ground_via_violations, 0,
        "the local ground via is present"
    );
    assert_eq!(
        clean.decoupling_power_layer_violations, 0,
        "cap power pad and IC power pin share F.Cu"
    );

    let dirty_lib = vec![cap_fp, ic_fp(LayerId(1))];
    let dirty = audit(&b, &comps, &dirty_lib, &DesignRules::holohv());
    assert_eq!(
        dirty.decoupling_ground_via_violations, 0,
        "the failing case is not missing its ground return via"
    );
    assert_eq!(
        dirty.decoupling_power_layer_violations, 1,
        "the cap power pad cannot reach the associated IC power pin without a layer change"
    );
}

#[test]
fn detects_oversized_decoupling_commutation_loop_area() {
    use crate::place::rotation::Rot;

    let mut b = board();
    let vdd = b.add_net("VDD_LOOP", NetClassKind::Power);
    let gnd = b.add_net("GND", NetClassKind::Ground);

    let lib = vec![
        FootprintDef::new(
            "U_LOOP",
            (Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Role::ActiveIc,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-2.0), Nm::from_mm(0.0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(2.0), Nm::from_mm(0.0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
            ],
        ),
        FootprintDef::new(
            "C_LOOP",
            (Nm::from_mm(2.0), Nm::from_mm(1.0)),
            Role::Decoupling,
            vec![
                PadDef {
                    offset: Point::new(Nm::from_mm(-0.5), Nm::from_mm(0.0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(0.5), Nm::from_mm(0.0)),
                    size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                    layers: vec![LayerId(0)],
                    power_pin: true,
                },
            ],
        ),
    ];
    let ic = Component {
        fp: 0,
        nets: vec![Some(vdd), Some(gnd)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let cap = |y_mm| Component {
        fp: 1,
        nets: vec![Some(vdd), Some(gnd)],
        refdes: "C1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(y_mm)),
            rot: Rot::R0,
        },
        assoc_ic: Some(0),
        locked: false,
        ..Default::default()
    };

    let near = vec![ic.clone(), cap(12.0)];
    let clean = audit(&b, &near, &lib, &DesignRules::holohv());
    assert_eq!(
        clean.decoupling_loop_area_violations, 0,
        "a 5 mm² commutation loop is inside the configured 10 mm² budget"
    );

    let far = vec![ic, cap(18.0)];
    let dirty = audit(&b, &far, &lib, &DesignRules::holohv());
    assert_eq!(
        dirty.decoupling_loop_area_violations, 1,
        "a 20 mm² commutation loop exceeds the configured loop-area budget"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "oversized decoupling commutation loops must reject clean-board selection"
    );
}

#[test]
fn detects_active_ic_power_pad_without_internal_plane() {
    use crate::place::rotation::Rot;

    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
    let vdd = b.add_net("VDD_CORE", NetClassKind::Power);
    let sig = b.add_net("GPIO0", NetClassKind::Signal);

    let lib = vec![FootprintDef::new(
        "U_THERMAL_POWER_PAD",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
                size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(1.5), Nm::from_mm(0.0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let comps = vec![Component {
        fp: 0,
        nets: vec![Some(vdd), Some(sig)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    b.zones.push(Zone {
        net: vdd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(12.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(12.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(8.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });

    let clean = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        clean.active_ic_power_plane_violations, 0,
        "a same-net internal plane under the active IC power pad satisfies the thermal-plane rule"
    );

    b.zones[0].net = sig;
    let dirty = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        dirty.active_ic_power_plane_violations, 1,
        "a signal zone under the power pad is not a same-net internal power plane"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "missing active-IC internal power-plane support must reject clean-board selection"
    );
}
