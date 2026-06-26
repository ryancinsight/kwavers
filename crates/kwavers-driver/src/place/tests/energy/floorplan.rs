//! Placement energy unit tests — Family A: floorplan/regional energy terms.
//!
//! All 13 tests here exercise `accumulate_floorplan` via the `regional`
//! field of the energy breakdown returned by `energy_fn`.

use super::*;

#[test]
fn regional_energy_separates_sensitive_ic_from_connector_emi_halo() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm::from_mm(8.0), Nm::from_mm(8.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        conn("J"),
    ];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(2.0),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let sig = NetId(10);
    let active = |x: f64| Component {
        fp: 0,
        nets: vec![Some(sig)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let connector = comp(1, "J1", 4.0, 10.0);

    let close_e = energy_fn(&[active(12.0), connector.clone()], &lib, &cfg, None);
    let separated_e = energy_fn(&[active(16.0), connector], &lib, &cfg, None);
    assert_eq!(
        close_e.regional, 2.5,
        "a 1.5 mm connector-to-IC courtyard gap violates the 4 mm EMI halo by 2.5 mm"
    );
    assert_eq!(
        separated_e.regional, 0.0,
        "a 5.5 mm connector-to-IC courtyard gap clears the 4 mm EMI halo"
    );
    assert!(
        separated_e.total < close_e.total,
        "regional placement must keep sensitive high-speed ICs away from connector EMI sources"
    );
}

#[test]
fn regional_energy_prefers_connector_ingress_toward_board_core() {
    let lib = vec![
        FootprintDef::new(
            "J",
            (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Role::Connector,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "U",
            (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(30.0)),
        margin: Nm::from_mm(0.0),
        thermal_spacing: Nm::from_mm(0.0),
        courtyard_clearance: Nm::from_mm(0.0),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let signal = NetId(10);
    let connector = Component {
        fp: 0,
        nets: vec![Some(signal)],
        refdes: "J1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let active = |x: f64, y: f64| Component {
        fp: 1,
        nets: vec![Some(signal)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };

    let inward = energy_fn(&[connector.clone(), active(14.0, 10.0)], &lib, &cfg, None);
    let transverse = energy_fn(&[connector, active(4.0, 20.0)], &lib, &cfg, None);
    assert_eq!(
        inward.regional, 0.0,
        "a connector-to-chip path pointing inward from the left board edge has no ingress penalty"
    );
    assert_eq!(
        transverse.regional, 1.0,
        "the same 10 mm path length adds a 1.0 connector-ingress penalty when the flow runs sideways"
    );
    assert!(
        inward.total < transverse.total,
        "regional placement must prefer smooth connector ingress toward the board core"
    );
}

#[test]
fn regional_energy_groups_local_functional_nets() {
    let lib = vec![FootprintDef::new(
        "U",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let board = (Nm::from_mm(40.0), Nm::from_mm(20.0));
    let cfg = PlaceConfig {
        board,
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let global = NetId(0);
    let region_a = NetId(1);
    let region_b = NetId(2);
    let mk = |refdes: &str, x: f64, y: f64, local: NetId| Component {
        fp: 0,
        nets: vec![Some(global), Some(local)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let grouped = vec![
        mk("A1", 6.0, 6.0, region_a),
        mk("A2", 9.0, 6.0, region_a),
        mk("B1", 28.0, 14.0, region_b),
        mk("B2", 31.0, 14.0, region_b),
    ];
    let interleaved = vec![
        mk("A1", 6.0, 6.0, region_a),
        mk("B1", 9.0, 6.0, region_b),
        mk("A2", 28.0, 14.0, region_a),
        mk("B2", 31.0, 14.0, region_b),
    ];

    let grouped_e = energy_fn(&grouped, &lib, &cfg, None);
    let interleaved_e = energy_fn(&interleaved, &lib, &cfg, None);
    assert_eq!(
        grouped_e.regional, 6.0,
        "two local 3 mm regions; the global net common to all parts is ignored"
    );
    assert!(
        grouped_e.regional < interleaved_e.regional,
        "regional placement should keep local functional groups compact"
    );
    assert!(
        grouped_e.total < interleaved_e.total,
        "regional weight must affect the placement objective"
    );
}

#[test]
fn regional_energy_penalizes_folded_local_signal_flow() {
    let lib = vec![FootprintDef::new(
        "U",
        (Nm(0), Nm(0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm(0),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let net_ab = NetId(10);
    let net_bc = NetId(20);
    let mk = |refdes: &str, x: f64, y: f64, nets: Vec<Option<NetId>>| Component {
        fp: 0,
        nets,
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let smooth = vec![
        mk("U1", 5.0, 10.0, vec![Some(net_ab), None]),
        mk("U2", 15.0, 10.0, vec![Some(net_ab), Some(net_bc)]),
        mk("U3", 25.0, 10.0, vec![None, Some(net_bc)]),
    ];
    let folded = vec![
        mk("U1", 5.0, 10.0, vec![Some(net_ab), None]),
        mk("U2", 15.0, 10.0, vec![Some(net_ab), Some(net_bc)]),
        mk("U3", 5.0, 10.0, vec![None, Some(net_bc)]),
    ];

    let smooth_e = energy_fn(&smooth, &lib, &cfg, None);
    let folded_e = energy_fn(&folded, &lib, &cfg, None);
    assert_eq!(
        smooth_e.regional, 20.0,
        "two 10 mm local regions with opposed flow through U2"
    );
    assert_eq!(
        folded_e.regional, 21.0,
        "the same two local regions plus one straight fold-back penalty at U2"
    );
    assert!(
        smooth_e.total < folded_e.total,
        "regional placement must prefer smooth unidirectional signal flow"
    );
}

#[test]
fn regional_energy_penalizes_orthogonal_local_signal_flow() {
    let lib = vec![FootprintDef::new(
        "U",
        (Nm(0), Nm(0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(30.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm(0),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let net_ab = NetId(10);
    let net_bc = NetId(20);
    let mk = |refdes: &str, x: f64, y: f64, nets: Vec<Option<NetId>>| Component {
        fp: 0,
        nets,
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let straight = vec![
        mk("U1", 5.0, 10.0, vec![Some(net_ab), None]),
        mk("U2", 15.0, 10.0, vec![Some(net_ab), Some(net_bc)]),
        mk("U3", 25.0, 10.0, vec![None, Some(net_bc)]),
    ];
    let dogleg = vec![
        mk("U1", 5.0, 10.0, vec![Some(net_ab), None]),
        mk("U2", 15.0, 10.0, vec![Some(net_ab), Some(net_bc)]),
        mk("U3", 15.0, 20.0, vec![None, Some(net_bc)]),
    ];
    let straight_e = energy_fn(&straight, &lib, &cfg, None);
    let dogleg_e = energy_fn(&dogleg, &lib, &cfg, None);
    assert_eq!(
        straight_e.regional, 20.0,
        "two 10 mm local regions with straight through-flow through U2"
    );
    assert_eq!(
        dogleg_e.regional, 21.0,
        "the two local regions plus one normalized right-angle dogleg penalty at U2"
    );
    assert!(
        straight_e.total < dogleg_e.total,
        "regional placement must prefer straight local signal flow over right-angle doglegs"
    );
}

#[test]
fn regional_energy_prefers_facing_main_chip_pads() {
    let lib = vec![FootprintDef::new(
        "U",
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(4.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.6), Nm::from_mm(0.6)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(1.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let net = Some(NetId(1));
    let left = Component {
        nets: vec![net],
        ..comp(0, "U1", 10.0, 10.0)
    };
    let facing = Component {
        nets: vec![net],
        placement: Placement {
            pos: Point::new(Nm::from_mm(30.0), Nm::from_mm(10.0)),
            rot: Rot::R180,
        },
        ..comp(0, "U2", 30.0, 10.0)
    };
    let away = Component {
        nets: vec![net],
        ..comp(0, "U2", 30.0, 10.0)
    };
    let facing_e = energy_fn(&[left.clone(), facing], &lib, &cfg, None);
    let away_e = energy_fn(&[left, away], &lib, &cfg, None);
    assert_eq!(
        facing_e.regional, 12.0,
        "facing pads leave a 12 mm main-chip pad-to-pad path"
    );
    assert_eq!(
        away_e.regional, 20.0,
        "away-facing pads leave a 20 mm main-chip pad-to-pad path"
    );
    assert!(
        facing_e.total < away_e.total,
        "regional placement must prefer shorter main-chip signal escapes"
    );
}

#[test]
fn regional_energy_groups_matching_power_ground_domains() {
    let lib = vec![FootprintDef::new(
        "U",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm::from_mm(0.5)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm::from_mm(0.5)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: true,
            },
            PadDef {
                offset: Point::new(Nm(0), Nm::from_mm(-0.5)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let gnd = NetId(0);
    let vdd_a = NetId(1);
    let vdd_b = NetId(2);
    let sig = NetId(3);
    let mk = |refdes: &str, x: f64, y: f64, vdd: NetId| Component {
        fp: 0,
        nets: vec![Some(vdd), Some(gnd), Some(sig)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let grouped = vec![
        mk("A1", 6.0, 6.0, vdd_a),
        mk("A2", 9.0, 6.0, vdd_a),
        mk("B1", 28.0, 14.0, vdd_b),
        mk("B2", 31.0, 14.0, vdd_b),
    ];
    let interleaved = vec![
        mk("A1", 6.0, 6.0, vdd_a),
        mk("B1", 9.0, 6.0, vdd_b),
        mk("A2", 28.0, 14.0, vdd_a),
        mk("B2", 31.0, 14.0, vdd_b),
    ];
    let grouped_e = energy_fn(&grouped, &lib, &cfg, None);
    let interleaved_e = energy_fn(&interleaved, &lib, &cfg, None);
    assert_eq!(grouped_e.regional, 6.0, "two matching VCC/GND rail-domain regions stay compact while common signal/ground nets are ignored as globals");
    assert!(
        grouped_e.regional < interleaved_e.regional,
        "regional placement should group components with matching VCC/GND domains"
    );
    assert!(
        grouped_e.total < interleaved_e.total,
        "rail-domain regional grouping must affect the placement objective"
    );
}

#[test]
fn regional_energy_groups_associated_support_components_with_main_ic() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "Y",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Role::Passive,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(50.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let mk = |fp: usize, refdes: &str, x: f64, assoc_ic: Option<usize>| Component {
        fp,
        nets: vec![None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic,
        locked: false,
        ..Default::default()
    };
    let grouped = vec![
        mk(0, "U1", 8.0, None),
        mk(0, "U2", 34.0, None),
        mk(1, "Y1", 11.0, Some(0)),
        mk(1, "Y2", 37.0, Some(1)),
    ];
    let swapped_support = vec![
        mk(0, "U1", 8.0, None),
        mk(0, "U2", 34.0, None),
        mk(1, "Y1", 37.0, Some(0)),
        mk(1, "Y2", 11.0, Some(1)),
    ];
    let grouped_e = energy_fn(&grouped, &lib, &cfg, None);
    let swapped_e = energy_fn(&swapped_support, &lib, &cfg, None);
    assert_eq!(
        grouped_e.regional, 6.0,
        "two associated IC/support regions each span 3 mm"
    );
    assert_eq!(swapped_e.regional, 60.0, "swapping support parts makes each associated region span 29 mm and intrude into the other IC region");
    assert!(
        grouped_e.total < swapped_e.total,
        "regional placement must keep associated support components with their main IC"
    );
}

#[test]
fn regional_energy_pulls_surge_suppressor_to_incoming_connector() {
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
            "D",
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
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let incoming = NetId(7);
    let mk = |fp: usize, refdes: &str, x: f64| Component {
        fp,
        nets: vec![Some(incoming)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let near = vec![mk(0, "J1", 4.0), mk(1, "D1", 5.5)];
    let far = vec![mk(0, "J1", 4.0), mk(1, "D1", 18.0)];
    let near_e = energy_fn(&near, &lib, &cfg, None);
    let far_e = energy_fn(&far, &lib, &cfg, None);
    assert_eq!(
        near_e.regional, 1.5,
        "the only regional penalty is the 1.5 mm connector-to-suppressor distance"
    );
    assert_eq!(far_e.regional, 14.0, "the same incoming net is common to all components here, so local-net regional grouping is inactive");
    assert!(
        near_e.total < far_e.total,
        "surge suppressors must be placed close to the incoming connector"
    );
}

#[test]
fn regional_energy_pulls_crystal_to_associated_ic_clock_pin() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm(0), Nm(0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "Y",
            (Nm(0), Nm(0)),
            Role::Passive,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(30.0), Nm::from_mm(20.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm(0),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let clk = NetId(10);
    let active = Component {
        fp: 0,
        nets: vec![Some(clk)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let oscillator = |x: f64| Component {
        fp: 1,
        nets: vec![Some(clk)],
        refdes: "Y1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: Some(0),
        locked: false,
        ..Default::default()
    };
    let near = energy_fn(&[active.clone(), oscillator(11.0)], &lib, &cfg, None);
    let far = energy_fn(&[active, oscillator(18.0)], &lib, &cfg, None);
    assert_eq!(
        near.regional, 1.0,
        "the two-component fixture isolates the 1 mm oscillator route distance"
    );
    assert_eq!(
        far.regional, 8.0,
        "the same isolated oscillator route is 8 mm when the crystal is moved away"
    );
    assert!(
        near.total < far.total,
        "regional placement must keep clock-source components near their associated IC pins"
    );
}

#[test]
fn regional_energy_penalizes_foreign_components_inside_a_functional_block() {
    let lib = vec![FootprintDef::new(
        "U",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(40.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let region_a = NetId(10);
    let unrelated = NetId(20);
    let mk = |refdes: &str, x: f64, y: f64, net: NetId| Component {
        fp: 0,
        nets: vec![Some(net)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let outside = vec![
        mk("A1", 8.0, 8.0, region_a),
        mk("A2", 20.0, 8.0, region_a),
        mk("A3", 8.0, 20.0, region_a),
        mk("X1", 30.0, 30.0, unrelated),
    ];
    let inside = vec![
        mk("A1", 8.0, 8.0, region_a),
        mk("A2", 20.0, 8.0, region_a),
        mk("A3", 8.0, 20.0, region_a),
        mk("X1", 14.0, 14.0, unrelated),
    ];
    let outside_e = energy_fn(&outside, &lib, &cfg, None);
    let inside_e = energy_fn(&inside, &lib, &cfg, None);
    assert_eq!(
        outside_e.regional, 24.0,
        "the A-region HPWL is 12 mm by 12 mm with no foreign intrusion"
    );
    assert_eq!(inside_e.regional, 33.0, "the foreign component at the region center adds 1 mm plus 6 mm exit depth and 2 mm package intrusion");
    assert!(
        outside_e.total < inside_e.total,
        "regional weight must reject a component placed in the middle of another block"
    );
}

#[test]
fn regional_energy_penalizes_foreign_courtyard_intrusion() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "WIDE",
            (Nm::from_mm(4.0), Nm::from_mm(2.0)),
            Role::Passive,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(40.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let region_a = NetId(10);
    let unrelated = NetId(20);
    let mk = |fp: usize, refdes: &str, x: f64, y: f64, net: NetId| Component {
        fp,
        nets: vec![Some(net)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let outside = vec![
        mk(0, "A1", 8.0, 8.0, region_a),
        mk(0, "A2", 20.0, 8.0, region_a),
        mk(0, "A3", 8.0, 20.0, region_a),
        mk(1, "X1", 25.0, 14.0, unrelated),
    ];
    let intruding = vec![
        mk(0, "A1", 8.0, 8.0, region_a),
        mk(0, "A2", 20.0, 8.0, region_a),
        mk(0, "A3", 8.0, 20.0, region_a),
        mk(1, "X1", 22.5, 14.0, unrelated),
    ];
    let outside_e = energy_fn(&outside, &lib, &cfg, None);
    let intruding_e = energy_fn(&intruding, &lib, &cfg, None);
    assert_eq!(
        outside_e.regional, 24.0,
        "the foreign component is outside the functional block envelope"
    );
    assert_eq!(intruding_e.regional, 25.5, "the foreign center is outside the block but its package enters the block envelope by 0.5 mm");
    assert!(
        outside_e.total < intruding_e.total,
        "regional placement must reject package-body intrusion, not only center intrusion"
    );
}

#[test]
fn regional_energy_keeps_power_circuitry_out_of_signal_region_halo() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "BUCK",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Role::Power,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: true,
            }],
        ),
    ];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(40.0), Nm::from_mm(30.0)),
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(1.0),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 0.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 1.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let signal = NetId(10);
    let power = NetId(20);
    let mk_signal = |refdes: &str, x: f64, y: f64| Component {
        fp: 0,
        nets: vec![Some(signal)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let mk_power = |x: f64| Component {
        fp: 1,
        nets: vec![Some(power)],
        refdes: "U_PWR".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let isolated = vec![
        mk_signal("U1", 8.0, 10.0),
        mk_signal("U2", 20.0, 10.0),
        mk_power(24.1),
    ];
    let encroaching = vec![
        mk_signal("U1", 8.0, 10.0),
        mk_signal("U2", 20.0, 10.0),
        mk_power(23.0),
    ];
    let isolated_e = energy_fn(&isolated, &lib, &cfg, None);
    let encroaching_e = energy_fn(&encroaching, &lib, &cfg, None);
    assert_eq!(isolated_e.regional, 24.0, "the signal region spans 12 mm, the main-chip pad path adds 12 mm, and the power package remains outside the 2 mm isolation halo");
    assert_eq!(encroaching_e.regional, 24.5, "the signal region and main-chip pad path are unchanged, and the power package enters half of the 2 mm halo");
    assert!(isolated_e.total < encroaching_e.total, "regional placement must keep high-current power circuitry outside signal-region isolation halos");
}
