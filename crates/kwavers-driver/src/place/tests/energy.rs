//! Placement energy and annealing tests.
//!
//! Section A — Tests lifted from src/place/mod.rs::tests (annealing + energy).

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Section A — Tests lifted from src/place/mod.rs::tests (annealing + energy)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn annealing_respects_footprint_rotation_policy() {
    let lib = vec![
        ic("U").with_rotation_policy(RotationPolicy::Fixed),
        FootprintDef::new(
            "C",
            (Nm::from_mm(1.0), Nm::from_mm(2.0)),
            Role::Decoupling,
            vec![PadDef {
                offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
    ];
    let board = (Nm::from_mm(30.0), Nm::from_mm(30.0));
    let cfg = PlaceConfig {
        board,
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights::default(),
        ..Default::default()
    };
    let params = AnnealParams {
        steps: 1,
        rot_prob: 1.0,
        ..Default::default()
    };
    let mut comps = vec![
        Component {
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R90,
            },
            ..comp(0, "U1", 10.0, 10.0)
        },
        Component {
            placement: Placement {
                pos: Point::new(Nm::from_mm(20.0), Nm::from_mm(20.0)),
                rot: Rot::R90,
            },
            ..comp(1, "C1", 20.0, 20.0)
        },
    ];

    anneal(&mut comps, &lib, &cfg, &[0], &params, None);
    anneal(&mut comps, &lib, &cfg, &[1], &params, None);

    assert_eq!(
        comps[0].placement.rot,
        Rot::R90,
        "fixed IC orientation must preserve the floorplan rotation"
    );
    assert_eq!(
        comps[1].placement.rot,
        Rot::R270,
        "decoupling rotation may flip only 180 degrees from the floorplanned axis"
    );
}

#[test]
fn utilization_energy_rewards_board_coverage() {
    let lib = vec![FootprintDef::new(
        "P",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let board = (Nm::from_mm(40.0), Nm::from_mm(40.0));
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
            utilization: 1.0,
            alignment: 0.0,
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let clustered = vec![
        comp(0, "P1", 19.5, 19.5),
        comp(0, "P2", 20.5, 19.5),
        comp(0, "P3", 19.5, 20.5),
        comp(0, "P4", 20.5, 20.5),
    ];
    let spread = vec![
        comp(0, "P1", 8.0, 8.0),
        comp(0, "P2", 32.0, 8.0),
        comp(0, "P3", 8.0, 32.0),
        comp(0, "P4", 32.0, 32.0),
    ];

    let clustered_e = energy_fn(&clustered, &lib, &cfg, None);
    let spread_e = energy_fn(&spread, &lib, &cfg, None);
    assert!(
        spread_e.utilization < clustered_e.utilization,
        "spread placement should leave less empty board: spread={} clustered={}",
        spread_e.utilization,
        clustered_e.utilization
    );
    assert!(
        spread_e.total < clustered_e.total,
        "utilization weight must affect the optimization objective"
    );
}

#[test]
fn utilization_ignores_locked_connectors_for_board_coverage() {
    let lib = vec![
        FootprintDef::new(
            "P",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Role::Passive,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        conn("J"),
    ];
    let board = (Nm::from_mm(40.0), Nm::from_mm(40.0));
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
            utilization: 1.0,
            alignment: 0.0,
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let mut with_locked_connector = vec![comp(0, "P1", 10.0, 10.0), comp(0, "P2", 12.0, 10.0)];
    let mut connector = comp(1, "J1", 32.0, 32.0);
    connector.locked = true;
    with_locked_connector.push(connector);
    let without_connector = vec![comp(0, "P1", 10.0, 10.0), comp(0, "P2", 12.0, 10.0)];

    let with_e = energy_fn(&with_locked_connector, &lib, &cfg, None);
    let without_e = energy_fn(&without_connector, &lib, &cfg, None);

    assert_eq!(
        with_e.utilization, without_e.utilization,
        "locked connector must not reduce functional board-coverage penalty"
    );
}

#[test]
fn alignment_energy_rewards_common_package_axis() {
    let lib = vec![FootprintDef::new(
        "R",
        (Nm::from_mm(2.0), Nm::from_mm(1.0)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(0.5), Nm(0)),
            size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let board = (Nm::from_mm(30.0), Nm::from_mm(20.0));
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
            alignment: 1.0,
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let aligned = vec![
        comp(0, "R1", 8.0, 8.0),
        Component {
            placement: Placement {
                pos: Point::new(Nm::from_mm(14.0), Nm::from_mm(8.0)),
                rot: Rot::R180,
            },
            ..comp(0, "R2", 14.0, 8.0)
        },
    ];
    let crossed = vec![
        comp(0, "R1", 8.0, 8.0),
        Component {
            placement: Placement {
                pos: Point::new(Nm::from_mm(14.0), Nm::from_mm(8.0)),
                rot: Rot::R90,
            },
            ..comp(0, "R2", 14.0, 8.0)
        },
    ];

    let aligned_e = energy_fn(&aligned, &lib, &cfg, None);
    let crossed_e = energy_fn(&crossed, &lib, &cfg, None);
    assert_eq!(aligned_e.alignment, 0.0, "0/180 share one placement axis");
    assert_eq!(
        crossed_e.alignment, 1.0,
        "0/90 on identical footprints is one axis mismatch"
    );
    assert!(
        aligned_e.total < crossed_e.total,
        "alignment weight must affect the placement objective"
    );
}

#[test]
fn airflow_blockage_penalizes_connector_in_hot_device_corridor() {
    let lib = vec![ic("U"), conn("J")];
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
            airflow_blockage: 1.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let blocked = vec![comp(0, "U1", 10.0, 10.0), comp(1, "J1", 5.0, 10.0)];
    let open = vec![comp(0, "U1", 10.0, 10.0), comp(1, "J1", 30.0, 10.0)];

    let blocked_e = energy_fn(&blocked, &lib, &cfg, None);
    let open_e = energy_fn(&open, &lib, &cfg, None);
    assert_eq!(
        blocked_e.airflow_blockage, 1.0,
        "the connector intersects the nearest-edge cooling corridor to the active IC"
    );
    assert_eq!(
        open_e.airflow_blockage, 0.0,
        "the connector outside the cooling corridor leaves airflow unobstructed"
    );
    assert!(
        open_e.total < blocked_e.total,
        "airflow blockage weight must affect the placement objective"
    );
}

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

#[test]
fn flow_crossing_energy_penalizes_crossed_local_nets() {
    let lib = vec![FootprintDef::new(
        "P",
        (Nm::from_mm(1.0), Nm::from_mm(1.0)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(30.0), Nm::from_mm(20.0)),
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
            regional: 0.0,
            flow_crossing: 1.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let net_a = NetId(10);
    let net_b = NetId(20);
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
    let crossed = vec![
        mk("A1", 5.0, 10.0, net_a),
        mk("A2", 25.0, 10.0, net_a),
        mk("B1", 15.0, 5.0, net_b),
        mk("B2", 15.0, 15.0, net_b),
    ];
    let uncrossed = vec![
        mk("A1", 5.0, 10.0, net_a),
        mk("A2", 25.0, 10.0, net_a),
        mk("B1", 5.0, 5.0, net_b),
        mk("B2", 25.0, 5.0, net_b),
    ];
    let crossed_e = energy_fn(&crossed, &lib, &cfg, None);
    let uncrossed_e = energy_fn(&uncrossed, &lib, &cfg, None);
    assert_eq!(
        crossed_e.flow_crossing, 1.0,
        "orthogonal local net flight lines cross once"
    );
    assert_eq!(
        uncrossed_e.flow_crossing, 0.0,
        "parallel local net flight lines do not cross"
    );
    assert!(
        uncrossed_e.total < crossed_e.total,
        "placement energy must prefer smooth non-crossing signal flow"
    );
}

#[test]
fn channel_blockage_energy_penalizes_foreign_component_in_routing_corridor() {
    let lib = vec![FootprintDef::new(
        "P",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(30.0), Nm::from_mm(20.0)),
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
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 1.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let routed = NetId(10);
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
    let open_channel = vec![
        mk("A1", 5.0, 10.0, routed),
        mk("A2", 25.0, 10.0, routed),
        mk("X1", 15.0, 15.0, unrelated),
    ];
    let blocked_channel = vec![
        mk("A1", 5.0, 10.0, routed),
        mk("A2", 25.0, 10.0, routed),
        mk("X1", 15.0, 10.0, unrelated),
    ];
    let open_e = energy_fn(&open_channel, &lib, &cfg, None);
    let blocked_e = energy_fn(&blocked_channel, &lib, &cfg, None);
    assert_eq!(
        open_e.channel_blockage, 0.0,
        "foreign package away from the flight line leaves the routing channel open"
    );
    assert_eq!(
        blocked_e.channel_blockage, 1.0,
        "foreign package courtyard intersects the local-net flight-line channel"
    );
    assert!(
        open_e.total < blocked_e.total,
        "placement energy must preserve a routing channel between associated components"
    );
}

#[test]
fn termination_energy_pulls_resistor_to_active_pad() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm::from_mm(3.0), Nm::from_mm(3.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "R",
            (Nm::from_mm(1.0), Nm::from_mm(0.6)),
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
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 1.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let tx = NetId(10);
    let active = Component {
        fp: 0,
        nets: vec![Some(tx)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let resistor = |x: f64| Component {
        fp: 1,
        nets: vec![Some(tx)],
        refdes: "R1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let near = vec![active.clone(), resistor(11.0)];
    let far = vec![active, resistor(18.0)];
    let near_e = energy_fn(&near, &lib, &cfg, None);
    let far_e = energy_fn(&far, &lib, &cfg, None);
    assert_eq!(
        near_e.termination, 1.0,
        "near terminator is 1 mm from the active pad"
    );
    assert_eq!(
        far_e.termination, 8.0,
        "far terminator is 8 mm from the active pad"
    );
    assert!(
        near_e.total < far_e.total,
        "termination placement energy must prefer the resistor near the active IC pad"
    );
}

#[test]
fn annealing_pulls_termination_resistor_toward_active_pad() {
    let lib = vec![
        FootprintDef::new(
            "U",
            (Nm::from_mm(3.0), Nm::from_mm(3.0)),
            Role::ActiveIc,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        ),
        FootprintDef::new(
            "R",
            (Nm::from_mm(1.0), Nm::from_mm(0.6)),
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
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights {
            overlap: 0.0,
            edge: 0.0,
            periphery: 0.0,
            decoupling: 0.0,
            termination: 1.0,
            hpwl: 0.0,
            thermal: 0.0,
            airflow_blockage: 0.0,
            utilization: 0.0,
            alignment: 0.0,
            regional: 0.0,
            flow_crossing: 0.0,
            channel_blockage: 0.0,
            ic_spread: 0.0,
            isolation_drift: 0.0,
            mech_keepout: 0.0,
        },
        ..Default::default()
    };
    let tx = NetId(10);
    let mut comps = vec![
        Component {
            fp: 0,
            nets: vec![Some(tx)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: true,
            ..Default::default()
        },
        Component {
            fp: 1,
            nets: vec![Some(tx)],
            refdes: "R1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(22.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        },
    ];
    let before = energy_fn(&comps, &lib, &cfg, None);
    let params = AnnealParams {
        steps: 500,
        t0: 1.0,
        cooling: 0.995,
        step_mm: 3.0,
        rot_prob: 0.0,
        seed: 0xA11C_EE55,
    };
    let after = anneal(&mut comps, &lib, &cfg, &[1], &params, None);
    assert_eq!(
        before.termination, 12.0,
        "test starts with the terminator 12 mm from the active pad"
    );
    assert!(
        after.termination < 2.0,
        "termination force must move the resistor inside the 2 mm placement budget, got {:.3} mm",
        after.termination
    );
    assert!(
        after.total < before.total,
        "termination-guided annealing must reduce weighted placement energy"
    );
}

#[test]
fn annealing_removes_overlap_and_peripheralises_connectors() {
    let lib = vec![ic("HV7355"), conn("J")];
    let board = (Nm::from_mm(60.0), Nm::from_mm(40.0));
    let mut comps = vec![
        comp(0, "U1", 30.0, 20.0),
        comp(0, "U2", 31.0, 21.0),
        comp(1, "J1", 29.0, 19.0),
        comp(1, "J2", 30.0, 20.0),
    ];
    let cfg = PlaceConfig {
        board,
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(12.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights::default(),
        ..Default::default()
    };
    let before = energy_fn(&comps, &lib, &cfg, None);
    assert!(before.overlap > 1.0, "test must start with real overlap");

    let movable: Vec<usize> = (0..comps.len()).collect();
    let params = AnnealParams {
        steps: 12_000,
        ..Default::default()
    };
    let after = anneal(&mut comps, &lib, &cfg, &movable, &params, None);
    assert!(
        after.overlap < 0.05,
        "annealing must remove courtyard overlap (got {} mm²)",
        after.overlap
    );
    assert!(after.total < before.total, "energy must decrease");
    let mean_conn_edge = (to_edge(&comps[2], &lib, board) + to_edge(&comps[3], &lib, board)) / 2.0;
    let mean_ic_edge = (to_edge(&comps[0], &lib, board) + to_edge(&comps[1], &lib, board)) / 2.0;
    assert!(mean_conn_edge < mean_ic_edge, "connectors (edge dist {mean_conn_edge:.1}) must be more peripheral than ICs ({mean_ic_edge:.1})");
    assert!(
        after.edge < 0.01,
        "no component may cross the edge margin (got {})",
        after.edge
    );
}

#[test]
fn congestion_field_pulls_components_to_quiet_regions() {
    let spec_res = GridSpec::cover(Nm::from_mm(40.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 1);
    let lib = vec![ic("U")];
    let board = (Nm::from_mm(40.0), Nm::from_mm(20.0));
    let cfg = PlaceConfig {
        board,
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(10.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights::default(),
        ..Default::default()
    };
    let spec = spec_res.unwrap();
    let mut per_column = vec![0.0f32; spec.nx * spec.ny];
    for iy in 0..spec.ny {
        for ix in 0..spec.nx {
            if spec.point_of(ix, iy).x.to_mm() < 20.0 {
                per_column[iy * spec.nx + ix] = 200.0;
            }
        }
    }
    let cg = CongestionField {
        spec,
        per_column,
        weight: 5.0,
    };
    let mut comps = vec![comp(0, "U1", 6.0, 10.0)];
    let params = AnnealParams {
        steps: 6_000,
        ..Default::default()
    };
    anneal(&mut comps, &lib, &cfg, &[0], &params, Some(&cg));
    assert!(
        comps[0].placement.pos.x.to_mm() > 20.0,
        "congestion must push the component into the quiet right half (got x={:.1})",
        comps[0].placement.pos.x.to_mm()
    );
}

#[test]
fn deterministic_for_a_given_seed() {
    let lib = vec![ic("U")];
    let board = (Nm::from_mm(40.0), Nm::from_mm(40.0));
    let cfg = PlaceConfig {
        board,
        margin: Nm::from_mm(1.0),
        thermal_spacing: Nm::from_mm(10.0),
        courtyard_clearance: Nm::from_mm(0.5),
        weights: PlaceWeights::default(),
        ..Default::default()
    };
    let params = AnnealParams {
        steps: 2_000,
        ..Default::default()
    };
    let run = || {
        let mut c = vec![comp(0, "U1", 10.0, 10.0), comp(0, "U2", 11.0, 11.0)];
        anneal(&mut c, &lib, &cfg, &[0, 1], &params, None);
        (c[0].placement, c[1].placement)
    };
    assert_eq!(run(), run(), "same seed must give identical placement");
}

#[test]
fn ic_spread_rewards_separation_of_same_fp_ics() {
    let lib = vec![ic("U")];
    let cfg = PlaceConfig {
        board: (Nm::from_mm(60.0), Nm::from_mm(40.0)),
        margin: Nm::from_mm(2.0),
        thermal_spacing: Nm::from_mm(5.0),
        courtyard_clearance: Nm::from_mm(1.0),
        weights: PlaceWeights::default(),
        ..Default::default()
    };
    let overlapping = vec![comp(0, "U1", 30.0, 20.0), comp(0, "U2", 30.0, 20.0)];
    let separated = vec![comp(0, "U1", 10.0, 10.0), comp(0, "U2", 50.0, 30.0)];
    let e_overlap = energy_fn(&overlapping, &lib, &cfg, None);
    let e_sep = energy_fn(&separated, &lib, &cfg, None);
    assert!(
        e_sep.ic_spread < e_overlap.ic_spread,
        "ic_spread must be lower for separated ICs: sep={:.3} overlap={:.3}",
        e_sep.ic_spread,
        e_overlap.ic_spread
    );
    assert!(
        e_sep.ic_spread > 0.0,
        "ic_spread must be non-zero even for well-separated ICs"
    );
}
