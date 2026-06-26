//! Placement energy unit tests — Family B: utilization, alignment, airflow,
//! flow-crossing, channel-blockage, termination, and IC-spread terms.
//!
//! Family A (floorplan/regional `accumulate_floorplan` tests) lives in the
//! `floorplan` sub-module.

pub(super) mod floorplan;

use super::*;

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
