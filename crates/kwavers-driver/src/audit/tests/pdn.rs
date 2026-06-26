use super::*;

#[test]
fn copper_imbalance_is_symmetric_pair_and_counts_planes() {
    use crate::board::{Zone, ZoneFill};
    // 4-layer board; a big plane polygon (a square) we can drop on chosen layers.
    let mk = |spec: GridSpec, gl: u16, vl: u16| {
        let mut b = Board::new(spec);
        let g = b.add_net("GND", NetClassKind::Ground);
        let v = b.add_net("VPP", NetClassKind::Hv);
        let sq = vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
        ];
        for (net, layer) in [(g, gl), (v, vl)] {
            b.zones.push(Zone {
                net,
                layer: LayerId(layer),
                polygon: sq.clone(),
                fill: ZoneFill::ThermalRelief,
            });
        }
        b
    };
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    // Planes on the symmetric inner pair (1,2): pairs (0,3) empty↔empty and (1,2) plane↔plane —
    // both matched ⇒ balanced.
    assert!(
        copper_imbalance(&mk(spec, 1, 2)) < 0.01,
        "symmetric plane placement is warp-balanced"
    );
    // Planes on adjacent layers (2,3): pair (0,3) empty↔plane and (1,2) empty↔plane — both
    // one-sided ⇒ imbalanced.
    assert!(
        copper_imbalance(&mk(spec, 2, 3)) > 0.9,
        "planes on a non-symmetric pair are imbalanced"
    );
}

#[test]
fn copper_imbalance_flags_single_layer_routing() {
    use crate::board::{NetClassKind, Track};
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    // All copper on layer 0 ⇒ strong imbalance (other layers empty).
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(9.0), Nm::from_mm(1.0)),
        width: Nm::from_mm(0.3),
        layer: LayerId(0),
        net: n,
    });
    assert!(
        copper_imbalance(&b) > 0.9,
        "single-layer copper must read as imbalanced"
    );
    // Mirror it onto layer 1 ⇒ balanced.
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(9.0), Nm::from_mm(1.0)),
        width: Nm::from_mm(0.3),
        layer: LayerId(1),
        net: n,
    });
    assert!(
        copper_imbalance(&b) < 0.01,
        "mirrored copper must read as balanced"
    );
}

#[test]
fn detects_decoupling_cap_without_local_ground_via() {
    use crate::board::{Via, ViaKind};
    use crate::place::component::Placement;

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
    use crate::board::{Via, ViaKind};
    use crate::place::component::Placement;

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
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};

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
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};

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

#[test]
fn charge_reservoir_violations_fire_on_under_provisioned_ic() {
    use crate::board::{LayerId, NetClassKind};
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};
    let ic_fp = FootprintDef::new(
        "U",
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ],
    )
    .with_i_dd_a(0.0); // per-path override below
    let cap_fp = FootprintDef::new(
        "C",
        (Nm::from_mm(2.0), Nm::from_mm(1.0)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ],
    )
    .with_capacitance_f(100e-12); // 100 pF → I_supply = 100e-12 * 3.3 / 5e-9 = 0.066 A

    let mk = |ic_i_dd: f64| {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut b = Board::new(spec);
        let vcc = b.add_net("+3V3", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        // IC has VCC on pad 0, GND on pad 1.
        let mut pwic_fp = ic_fp.clone();
        pwic_fp.i_dd_a = ic_i_dd;
        let pwic_lib = vec![pwic_fp, cap_fp.clone()];
        let ic = Component {
            fp: 0,
            nets: vec![Some(vcc), Some(gnd)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let cap = Component {
            fp: 1,
            nets: vec![Some(vcc), Some(gnd)],
            refdes: "C1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(9.5), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: Some(0),
            locked: false,
            ..Default::default()
        };
        (b, vec![ic, cap], pwic_lib)
    };

    // Path 1 — under-provisioned: I_dd = 1.0 A, single 100 pF cap delivers 0.066 A.
    let (b, comps, lib) = mk(1.0);
    let r = audit(&b, &comps, &lib, &DesignRules::holohv());
    assert_eq!(
        r.charge_reservoir_violations, 1,
        "I_dd = 1.0 A and one 100 pF cap (I_supply = 0.066 A) must underflow"
    );
    assert!(
        !r.hard_drc_clean(),
        "hard_drc_clean must trip on a single charge-reservoir violation"
    );
    assert!(
        r.risk_score >= 20.0,
        "risk_score must fold the charge-reservoir tier (20.0/violation)"
    );
    let ic_x_nm: i64 = Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0;
    assert!(
        r.hotspots.iter().any(|p| p.x.0 == ic_x_nm),
        "hotspot must mark the IC's VPP pad at x = 8 mm − 2 mm = 6 mm, not the cap"
    );

    // Path 2 — vacuous: i_dd_a = 0.0 ⇒ no rating set, detector skips silently.
    let (b2, comps2, lib2) = mk(0.0);
    let r2 = audit(&b2, &comps2, &lib2, &DesignRules::holohv());
    assert_eq!(
        r2.charge_reservoir_violations, 0,
        "an IC with no datasheet I_dd must read vacuous (the validate-style pattern)"
    );

    // Path 3 — just balanced: i_dd_a = 0.066 A ⇒ supply exactly meets demand (no violation).
    let (b3, comps3, lib3) = mk(0.066);
    let r3 = audit(&b3, &comps3, &lib3, &DesignRules::holohv());
    assert_eq!(
        r3.charge_reservoir_violations, 0,
        "exactly balanced (I_supply ≋ I_dd) must not flag a violation"
    );

    // Path 4 — with no caps: i_supply = 0, but i_dd > 0 ⇒ still a violation.
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b4 = Board::new(spec);
    let vcc = b4.add_net("+3V3", NetClassKind::Power);
    let gnd = b4.add_net("GND", NetClassKind::Ground);
    let pads = vec![
        PadDef {
            offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: true,
        },
        PadDef {
            offset: Point::new(Nm::from_mm(2.0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: true,
        },
    ];
    let ic_only_fp = FootprintDef::new(
        "U",
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::ActiveIc,
        pads,
    )
    .with_i_dd_a(1.0);
    let lib4 = vec![ic_only_fp];
    let comps4 = vec![Component {
        fp: 0,
        nets: vec![Some(vcc), Some(gnd)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    let r4 = audit(&b4, &comps4, &lib4, &DesignRules::holohv());
    assert_eq!(
        r4.charge_reservoir_violations, 1,
        "an IC with no associated cap must still violate — sum is 0, demand > 0"
    );
}

#[test]
fn charge_reservoir_violations_fire_on_under_provisioned_buck() {
    // Mirror of `charge_reservoir_violations_fire_on_under_provisioned_ic`:
    // same `assoc_ic`-tied `Role::Decoupling` cap pool and
    // `I_supply = C · dV / dt` math (dv = 3.3 V, dt = 5 ns from `holohv()`);
    // the only delta is `Role::Power` (buck converter) at the consumer side.
    // Vacuous semantics (`i_dd_a ≤ 0.0`), the `+ 1e-12` slack, the first
    // power-pin pad hotspot, and `risk_score`-fold weight all match by construction
    // — the only new coverage is that the detector recognises the buck.
    use crate::board::{LayerId, NetClassKind};
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::{Rot};
    let buck_fp = FootprintDef::new(
        "BUCK",
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::Power,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(2.0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ],
    );
    let cap_fp = FootprintDef::new(
        "C",
        (Nm::from_mm(2.0), Nm::from_mm(1.0)),
        Role::Decoupling,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
        ],
    )
    .with_capacitance_f(100e-12); // 100 pF → I_supply = 100e-12 · 3.3 / 5e-9 = 0.066 A
    let mk = |buck_i_dd_a: f64, cap_c_f: f64| {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut b = Board::new(spec);
        let vin = b.add_net("+12V_IN", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);
        // Buck has V_IN on pad 0 (power_pin = true), GND on pad 1.
        let mut ppwr_fp = buck_fp.clone();
        ppwr_fp.i_dd_a = buck_i_dd_a;
        let mut pcap_fp = cap_fp.clone();
        pcap_fp.capacitance_f = cap_c_f;
        let pwic_lib = vec![ppwr_fp, pcap_fp];
        let buck = Component {
            fp: 0,
            nets: vec![Some(vin), Some(gnd)],
            refdes: "BUCK1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        };
        let cap = Component {
            fp: 1,
            nets: vec![Some(vin), Some(gnd)],
            refdes: "C1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(9.5), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: Some(0), // ties the cap to the buck at fp 0.
            locked: false,
            ..Default::default()
        };
        (b, vec![buck, cap], pwic_lib)
    };

    // Path 1 — sized: cap 100 pF, buck I_dd = 0.066 A (≈ I_supply, exact
    // balance after the detector's `+ 1e-12` slack). Demand sat.
    let (b1, comps1, lib1) = mk(0.066, 100e-12);
    let r1 = audit(&b1, &comps1, &lib1, &DesignRules::holohv());
    assert_eq!(
        r1.charge_reservoir_violations, 0,
        "buck with I_dd == 0.066 A and one 100 pF cap (I_supply = 0.066 A) must not violate",
    );
    // No `hard_drc_clean()` assertion here — the bare 20 mm × 20 mm fixture
    // has no tracks, so other audit clauses (clearance/dangling/...) still
    // legitimately flag the empty-board baseline. The relevant guarantee
    // is just the charge-reservoir count, matching the IC test.

    // Path 2 — vacuous: i_dd_a = 0.0 ⇒ no datasheet rating set; the
    // detector silently skips the buck (the validate-style vacuous pattern
    // that mirrors the IC test's Path 2).
    let (b2, comps2, lib2) = mk(0.0, 100e-12);
    let r2 = audit(&b2, &comps2, &lib2, &DesignRules::holohv());
    assert_eq!(
        r2.charge_reservoir_violations, 0,
        "a buck with no datasheet I_dd must read vacuous — detector skips silently",
    );

    // Path 3 — under-sized caps: same I_dd = 0.066 A, but cap 1 pF
    // (I_supply = 1e-12 · 3.3 / 5e-9 = 6.6e-4 A ≪ 0.066 A).
    let (b3, comps3, lib3) = mk(0.066, 1e-12);
    let r3 = audit(&b3, &comps3, &lib3, &DesignRules::holohv());
    assert_eq!(
        r3.charge_reservoir_violations, 1,
        "buck with I_dd = 0.066 A and one 1 pF cap must underflow (sum = 0.00066 A)",
    );
    assert!(
        !r3.hard_drc_clean(),
        "hard_drc_clean must trip on a single buck charge-reservoir violation",
    );
    assert!(
        r3.risk_score >= 20.0,
        "risk_score must fold the charge-reservoir tier (20.0/violation)",
    );
    // Hotspot at the buck's first power-pin pad: pad 0 at offset (-2.0, 0) mm
    // from the buck footprint centre (8.0, 10.0) mm → hotspot
    // p.x.0 == Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0 (i.e. 6.0 mm), not the cap.
    let buck_vin_x_nm: i64 = Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0;
    assert!(
        r3.hotspots.iter().any(|p| p.x.0 == buck_vin_x_nm),
        "hotspot must mark the buck's V_IN pad at x = 6.0 mm, not the cap",
    );

    // Path 4 — no caps: i_supply = 0 (no `Role::Decoupling` entries tied
    // to the buck via `assoc_ic`), but buck I_dd > 0 ⇒ the detector still
    // trips on its own. Mirrors the IC test's Path 4: `lib` has only the
    // consumer footprint; `comps` has only the consumer component (no cap
    // with `assoc_ic = Some(0)`). Sum = 0, demand > 0 ⇒ underflow.
    let spec4 =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b4 = Board::new(spec4);
    let vin4 = b4.add_net("+12V_IN", NetClassKind::Power);
    let gnd4 = b4.add_net("GND", NetClassKind::Ground);
    let pads_buck = vec![
        PadDef {
            offset: Point::new(Nm::from_mm(-2.0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: true,
        },
        PadDef {
            offset: Point::new(Nm::from_mm(2.0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: true,
        },
    ];
    let buck_only_fp = FootprintDef::new(
        "BUCK",
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::Power,
        pads_buck,
    )
    .with_i_dd_a(0.066); // demands 0.066 A with no caps ⇒ sum = 0 ⇒ underflow
    let lib4 = vec![buck_only_fp];
    let comps4 = vec![Component {
        fp: 0,
        nets: vec![Some(vin4), Some(gnd4)],
        refdes: "BUCK1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    let r4 = audit(&b4, &comps4, &lib4, &DesignRules::holohv());
    assert_eq!(
        r4.charge_reservoir_violations, 1,
        "buck with no associated cap must still violate — sum is 0, demand > 0",
    );
}

#[test]
fn charge_recycling_fires_on_nlevel_ic_without_cr_bus() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let board = Board::new(spec); // no CR_* nets

    let fp = FootprintDef::new(
        "MD1715-DB",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    );
    let comp = Component {
        refdes: "U1".into(),
        fp: 0,
        nets: vec![],
        placement: Placement {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let (count, pts) = detect_charge_recycling_violations_board(&board, &[comp], &[fp]);
    assert_eq!(
        count, 1,
        "one N-level IC without a CR bus net → 1 violation"
    );
    assert_eq!(pts.len(), 1);
}

#[test]
fn charge_recycling_passes_when_cr_net_present() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef};
use crate::place::rotation::{Rot};
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
    board.add_net("CHR_BUS", NetClassKind::Signal);

    let fp = FootprintDef::new(
        "MAX14815-AAE",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    );
    let comp = Component {
        refdes: "U1".into(),
        fp: 0,
        nets: vec![],
        placement: Placement {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let (count, _) = detect_charge_recycling_violations_board(&board, &[comp], &[fp]);
    assert_eq!(
        count, 0,
        "N-level IC with a CR bus net present → no violation"
    );
}

#[test]
fn pulse_skip_fires_when_error_exceeds_tolerance() {
    let spec =
        GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
    // 4 TX nets — with 80% skip fraction, rms_err = sqrt(0.8/4) = 0.447, >> 5% tol
    for i in 0..4 {
        board.add_net(format!("TX_{i}"), NetClassKind::Hv);
    }
    let mut rules = DesignRules::holohv();
    rules.max_skip_fraction = 0.8;
    rules.pressure_error_tol = 0.05;
    let (count, _) = detect_pulse_skip_violations(&board, &rules);
    assert_eq!(count, 1, "high skip fraction on few channels → violation");
}

#[test]
fn pulse_skip_passes_within_tolerance() {
    let spec =
        GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
    // 96 TX nets — with 20% skip: rms_err = sqrt(0.2/96) ≈ 0.046, just under 5% tol
    for i in 0..96 {
        board.add_net(format!("TX_{i}"), NetClassKind::Hv);
    }
    let mut rules = DesignRules::holohv();
    rules.max_skip_fraction = 0.2;
    rules.pressure_error_tol = 0.05;
    let (count, _) = detect_pulse_skip_violations(&board, &rules);
    assert_eq!(count, 0, "20% skip on 96 channels is within 5% error tol");
}
