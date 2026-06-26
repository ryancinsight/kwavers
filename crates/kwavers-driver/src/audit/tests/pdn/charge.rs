//! Charge-reservoir, charge-recycling, and pulse-skip PDN tests.

use super::*;
use crate::place::component::Placement;
use crate::place::footprint::PadDef;
use crate::place::rotation::Rot;

#[test]
fn charge_reservoir_violations_fire_on_under_provisioned_ic() {
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
    .with_i_dd_a(0.0);
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
    .with_capacitance_f(100e-12);

    let mk = |ic_i_dd: f64| {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut b = Board::new(spec);
        let vcc = b.add_net("+3V3", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);
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

    let (b2, comps2, lib2) = mk(0.0);
    let r2 = audit(&b2, &comps2, &lib2, &DesignRules::holohv());
    assert_eq!(
        r2.charge_reservoir_violations, 0,
        "an IC with no datasheet I_dd must read vacuous (the validate-style pattern)"
    );

    let (b3, comps3, lib3) = mk(0.066);
    let r3 = audit(&b3, &comps3, &lib3, &DesignRules::holohv());
    assert_eq!(
        r3.charge_reservoir_violations, 0,
        "exactly balanced (I_supply ≋ I_dd) must not flag a violation"
    );

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
    .with_capacitance_f(100e-12);
    let mk = |buck_i_dd_a: f64, cap_c_f: f64| {
        let spec =
            GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
        let mut b = Board::new(spec);
        let vin = b.add_net("+12V_IN", NetClassKind::Power);
        let gnd = b.add_net("GND", NetClassKind::Ground);
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
            assoc_ic: Some(0),
            locked: false,
            ..Default::default()
        };
        (b, vec![buck, cap], pwic_lib)
    };

    let (b1, comps1, lib1) = mk(0.066, 100e-12);
    let r1 = audit(&b1, &comps1, &lib1, &DesignRules::holohv());
    assert_eq!(
        r1.charge_reservoir_violations, 0,
        "buck with I_dd == 0.066 A and one 100 pF cap (I_supply = 0.066 A) must not violate",
    );

    let (b2, comps2, lib2) = mk(0.0, 100e-12);
    let r2 = audit(&b2, &comps2, &lib2, &DesignRules::holohv());
    assert_eq!(
        r2.charge_reservoir_violations, 0,
        "a buck with no datasheet I_dd must read vacuous — detector skips silently",
    );

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
    let buck_vin_x_nm: i64 = Nm::from_mm(8.0).0 - Nm::from_mm(2.0).0;
    assert!(
        r3.hotspots.iter().any(|p| p.x.0 == buck_vin_x_nm),
        "hotspot must mark the buck's V_IN pad at x = 6.0 mm, not the cap",
    );

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
    .with_i_dd_a(0.066);
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
    use crate::audit::detect_power::detect_charge_recycling_violations_board;
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let board = Board::new(spec);

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
    use crate::audit::detect_power::detect_charge_recycling_violations_board;
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
    use crate::audit::detect_power::detect_pulse_skip_violations;
    let spec =
        GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
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
    use crate::audit::detect_power::detect_pulse_skip_violations;
    let spec =
        GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
    for i in 0..96 {
        board.add_net(format!("TX_{i}"), NetClassKind::Hv);
    }
    let mut rules = DesignRules::holohv();
    rules.max_skip_fraction = 0.2;
    rules.pressure_error_tol = 0.05;
    let (count, _) = detect_pulse_skip_violations(&board, &rules);
    assert_eq!(count, 0, "20% skip on 96 channels is within 5% error tol");
}
