//! Slice-wide tests for the `place` slice.
//!
//! Phase 2c consolidated the previously-inline `mod tests { ... }` blocks of
//! `src/place/{mod, footprint, footprint_import, component, symbol_import}.rs`
//! into a single `crate::place::tests` module. The pattern matches the Phase 2a
//! `cost::tests` / Phase 2b `route::tests` migration: one slice-wide test file
//! collects the topology tests + parser pinning tests + integration tests so an
//! external reviewer can grep `tests.rs` once and see the entire place-slice
//! behaviour contract.
//!
//! The S-expression kernel (`Sexpr`, `parse_sexpr`, `child`, `num`, `xyz_child`) now lives at
//! `crate::place::sexpr` (`pub(crate)`) — the SSOT shared with `io::pcb_parse`. The byte-tracking
//! pinning tests here (`parse_sexpr_unclosed_*`,
//! `parse_sexpr_unicode_byte_offset_differs_from_char_offset`,
//! `imported_model_offset_is_recentered_with_pads`,
//! `imports_model_offset_and_rotation`) reach them via `use crate::place::sexpr::*`.

use crate::board::{LayerId, NetId};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::sexpr::{child, parse_sexpr, xyz_child};
use crate::place::{
    anneal, energy, import_kicad_mod, import_symbol_pinmap, AnnealParams, Component,
    CongestionField, FootprintDef, PadDef, PinMap, PlaceConfig, PlaceWeights, Placement, Rect,
    Role, Rot, RotationPolicy,
};

// ─── Fixtures (lifted from inline `mod tests` blocks of mod.rs / footprint.rs / component.rs) ───

fn ic(name: &str) -> FootprintDef {
    FootprintDef::new(
        name,
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(4.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.6), Nm::from_mm(0.6)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    )
}

fn conn(name: &str) -> FootprintDef {
    FootprintDef::new(
        name,
        (Nm::from_mm(5.0), Nm::from_mm(16.0)),
        Role::Connector,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )
}

fn comp(fp: usize, refdes: &str, x: f64, y: f64) -> Component {
    Component {
        fp,
        nets: vec![None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }
}

/// Distance (mm) from a component's courtyard to the nearest board edge.
fn to_edge(c: &Component, lib: &[FootprintDef], board: (Nm, Nm)) -> f64 {
    let r = c.courtyard(lib);
    let w = board.0.to_mm();
    let h = board.1.to_mm();
    r.min
        .x
        .to_mm()
        .min(w - r.max.x.to_mm())
        .min(r.min.y.to_mm())
        .min(h - r.max.y.to_mm())
}

/// One pad-row of an N-pad footprint at `pitch`, used to exercise the fine-pitch escape predicate.
fn row_fp(pitch_mm: f64, n: usize) -> FootprintDef {
    let pads = (0..n)
        .map(|k| PadDef {
            offset: Point::new(Nm::from_mm(k as f64 * pitch_mm), Nm(0)),
            size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
            layers: vec![LayerId(0)],
            power_pin: false,
        })
        .collect();
    FootprintDef::new(
        "row",
        (Nm::from_mm(n as f64 * pitch_mm), Nm::from_mm(1.0)),
        Role::ActiveIc,
        pads,
    )
}

fn lib() -> Vec<FootprintDef> {
    vec![FootprintDef::new(
        "U",
        (Nm::from_mm(8.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(3.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    )]
}


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

    let clustered_e = energy(&clustered, &lib, &cfg, None);
    let spread_e = energy(&spread, &lib, &cfg, None);
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

    let with_e = energy(&with_locked_connector, &lib, &cfg, None);
    let without_e = energy(&without_connector, &lib, &cfg, None);

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

    let aligned_e = energy(&aligned, &lib, &cfg, None);
    let crossed_e = energy(&crossed, &lib, &cfg, None);
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

    let blocked_e = energy(&blocked, &lib, &cfg, None);
    let open_e = energy(&open, &lib, &cfg, None);
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

    let close_e = energy(&[active(12.0), connector.clone()], &lib, &cfg, None);
    let separated_e = energy(&[active(16.0), connector], &lib, &cfg, None);
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

    let inward = energy(&[connector.clone(), active(14.0, 10.0)], &lib, &cfg, None);
    let transverse = energy(&[connector, active(4.0, 20.0)], &lib, &cfg, None);
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

    let grouped_e = energy(&grouped, &lib, &cfg, None);
    let interleaved_e = energy(&interleaved, &lib, &cfg, None);
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

    let smooth_e = energy(&smooth, &lib, &cfg, None);
    let folded_e = energy(&folded, &lib, &cfg, None);
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
    let straight_e = energy(&straight, &lib, &cfg, None);
    let dogleg_e = energy(&dogleg, &lib, &cfg, None);
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
    let facing_e = energy(&[left.clone(), facing], &lib, &cfg, None);
    let away_e = energy(&[left, away], &lib, &cfg, None);
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
    let grouped_e = energy(&grouped, &lib, &cfg, None);
    let interleaved_e = energy(&interleaved, &lib, &cfg, None);
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
    let grouped_e = energy(&grouped, &lib, &cfg, None);
    let swapped_e = energy(&swapped_support, &lib, &cfg, None);
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
    let near_e = energy(&near, &lib, &cfg, None);
    let far_e = energy(&far, &lib, &cfg, None);
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
    let near = energy(&[active.clone(), oscillator(11.0)], &lib, &cfg, None);
    let far = energy(&[active, oscillator(18.0)], &lib, &cfg, None);
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
    let outside_e = energy(&outside, &lib, &cfg, None);
    let inside_e = energy(&inside, &lib, &cfg, None);
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
    let outside_e = energy(&outside, &lib, &cfg, None);
    let intruding_e = energy(&intruding, &lib, &cfg, None);
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
    let isolated_e = energy(&isolated, &lib, &cfg, None);
    let encroaching_e = energy(&encroaching, &lib, &cfg, None);
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
    let crossed_e = energy(&crossed, &lib, &cfg, None);
    let uncrossed_e = energy(&uncrossed, &lib, &cfg, None);
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
    let open_e = energy(&open_channel, &lib, &cfg, None);
    let blocked_e = energy(&blocked_channel, &lib, &cfg, None);
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
    let near_e = energy(&near, &lib, &cfg, None);
    let far_e = energy(&far, &lib, &cfg, None);
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
    let before = energy(&comps, &lib, &cfg, None);
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
    let before = energy(&comps, &lib, &cfg, None);
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
    let e_overlap = energy(&overlapping, &lib, &cfg, None);
    let e_sep = energy(&separated, &lib, &cfg, None);
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

// ────────────────────────────────────────────────────────────────────────────
// Section B — Tests lifted from src/place/footprint.rs (rotation + role +
// footprint escape). These tests now resolve `Rot` + `RotationPolicy` via
// `crate::place::rotation::*` because the carve at Phase 2c extracted them out
// of `footprint.rs`.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn rotation_is_order_four_and_preserves_length() {
    let p = Point::new(Nm::from_mm(3.0), Nm::from_mm(1.0));
    assert_eq!(
        Rot::R90.apply(Rot::R90.apply(Rot::R90.apply(Rot::R90.apply(p)))),
        p
    );
    let r = Rot::R90.apply(p);
    assert_eq!(r, Point::new(Nm::from_mm(-1.0), Nm::from_mm(3.0)));
}

#[test]
fn quarter_turn_swaps_courtyard_axes() {
    let s = (Nm::from_mm(8.0), Nm::from_mm(3.0));
    assert_eq!(Rot::R0.apply_size(s), s);
    assert_eq!(Rot::R90.apply_size(s), (Nm::from_mm(3.0), Nm::from_mm(8.0)));
    assert_eq!(Rot::R180.apply_size(s), s);
}

#[test]
fn role_defaults_constrain_rotation() {
    assert_eq!(
        RotationPolicy::for_role(Role::ActiveIc),
        RotationPolicy::Fixed
    );
    assert_eq!(
        RotationPolicy::for_role(Role::Connector),
        RotationPolicy::Fixed
    );
    assert_eq!(
        RotationPolicy::for_role(Role::Decoupling),
        RotationPolicy::HalfTurn
    );
}

#[test]
fn half_turn_policy_preserves_the_floorplanned_axis() {
    assert_eq!(
        Rot::R90.next_allowed(Rot::R90, RotationPolicy::HalfTurn),
        Some(Rot::R270)
    );
    assert_eq!(
        Rot::R270.next_allowed(Rot::R90, RotationPolicy::HalfTurn),
        Some(Rot::R90)
    );
    assert_eq!(Rot::R90.next_allowed(Rot::R90, RotationPolicy::Fixed), None);
    assert_eq!(
        Rot::R90.next_allowed(Rot::R90, RotationPolicy::AnyRightAngle),
        Some(Rot::R180)
    );
}

#[test]
fn min_pad_pitch_is_the_nearest_pad_spacing() {
    assert_eq!(row_fp(0.5, 4).min_pad_pitch(), Some(Nm::from_mm(0.5)));
    assert_eq!(row_fp(0.5, 1).min_pad_pitch(), None);
}

#[test]
fn fine_pitch_triggers_escape_and_coarse_does_not() {
    let thresh = Nm::from_mm(0.7);
    assert!(
        row_fp(0.5, 8).needs_escape(thresh),
        "0.5 mm pitch must escape"
    );
    assert!(
        !row_fp(1.1, 2).needs_escape(thresh),
        "1.1 mm pitch routes on top"
    );
    let bga = FootprintDef::bga("bga", 4, 4, Nm::from_mm(0.8), &[]);
    assert!(bga.needs_escape(thresh), "an explicit BGA always escapes");
}

// ────────────────────────────────────────────────────────────────────────────
// Section C — Tests lifted from src/place/footprint_import.rs::tests
// (sexpr parser byte-tracking pinning + realshift vendor importer tests).
//
// The S-expression kernel (`Sexpr`, `parse_sexpr`, `child`, `num`, `xyz_child`) is now at
// `crate::place::sexpr` (`pub(crate)`); these tests reach it via `use crate::place::sexpr::*`.
// The DOCS path is preserved verbatim from the original inline module so the
// committed vendor files stay the differential oracle.
// ────────────────────────────────────────────────────────────────────────────

// Path to the committed vendor footprints, relative to the crate root.
const DOCS: &str = "docs/cad_models";

#[test]
fn imports_real_xc7a100t_fgg484_bga() {
    // The XC7A100T-2FGG484C AMD vendor footprint is a 484-ball FG(G)BGA, 22×22 grid (with corner
    // depopulation), 1.0 mm pitch. The geometry file gives each ball a letter-number designator
    // ("A1".."AB22"); `pad_index()` lets the netlist wire a net to a specific ball by name.
    let p =
        format!("{DOCS}/XC7A100T_2FGG484C/KiCADv6/footprints.pretty/FGG484ARTIX-7_AMD.kicad_mod");
    let fp = import_kicad_mod(&p, Role::ActiveIc, &["A2", "H8"]).unwrap();
    assert_eq!(fp.pads.len(), 484, "real FGG484 pad count");
    assert_eq!(fp.pad_names.len(), 484, "real FGG484 ball designators");
    assert!(fp.pad_index("A1").is_some(), "ball A1 exists");
    assert!(
        fp.pad_index("L11").is_some(),
        "ball L11 exists (centre row/col)"
    );
    assert!(
        fp.pad_index("W22").is_some(),
        "ball W22 exists (max row/col)"
    );
    assert!(fp.pads[fp.pad_index("A2").unwrap()].power_pin);
    assert!(fp.pads[fp.pad_index("H8").unwrap()].power_pin);
    assert!(fp.courtyard.0.to_mm() > 23.0 && fp.courtyard.1.to_mm() > 23.0);
}

#[test]
fn parses_nested_sexpr_with_quotes() {
    let s = parse_sexpr(r#"(footprint "A B" (pad "1" smd (at 1 2)))"#).unwrap();
    assert_eq!(s.head(), Some("footprint"));
    let pad = child(&s, "pad").unwrap();
    assert_eq!(pad.as_list().unwrap()[1].as_atom(), Some("1"));
}

// -------- Phase 1c polish: `Manifest::Parse { offset }` byte-tracking pinning tests --------
//
// These pin the parse_sexpr byte-position contract. The `parse_sexpr` loop iterates
// `char_indices().peekable()` (NOT `chars().enumerate()`); every `Manifest::Parse` carries
// the TRUE UTF-8 byte offset of the offending token, the EOF position when the input ran out,
// or an explicit byte position for nested unclosed forms. The tests use direct pattern
// matching on `crate::error::Error::Manifest(Manifest::Parse {..})` (mirroring the SSOT
// smoke tests at `src/error/manifest.rs::tests::io_at_matches_inline_construction`); the
// aggregator's `#[error(transparent)]` on `Error::Manifest` delegates `source()` straight to
// the inner `Manifest::Parse::source()` (returning `None`, since `Parse` has no `#[source]`
// field), so source-chain walking would NOT find the `Manifest` envelope.

#[test]
fn parse_sexpr_unclosed_paren_offset_points_at_offender() {
    let input = "a)";
    let err = parse_sexpr(input).expect_err("rogue closer must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(*offset, 1, "byte offset must point at the offending `)`");
            assert!(
                message.contains("unexpected closing paren"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("rogue `)` must produce Manifest::Parse; got {err:?}"),
    }
    let s = err.to_string();
    assert!(s.contains("near byte 1"), "Display carries the offset: {s}");
}

#[test]
fn parse_sexpr_unclosed_string_reports_eof_offset() {
    let input = r#"(pad "abc"#;
    let err = parse_sexpr(input).expect_err("unclosed quote must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset,
                input.len(),
                "offset must be src.len() (EOF after the unclosed quote): got {offset}"
            );
            assert!(
                message.contains("unclosed string literal"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("unclosed quote must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn parse_sexpr_eof_before_top_level_reports_input_len() {
    let input = "(footprint"; // 10 bytes, no closing `)`
    let err = parse_sexpr(input).expect_err("EOF before close must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset,
                input.len(),
                "offset must be src.len() (input exhausted): got {offset}"
            );
            assert!(
                message.contains("input ended before top-level s-expression closed"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("EOF-before-close must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn parse_sexpr_unicode_byte_offset_differs_from_char_offset() {
    let input = "\u{03bc})"; // 3 bytes: [0xCE, 0xBC, 0x29]
    assert_eq!(input.len(), 3, "µ must be 2 bytes in UTF-8");
    let err = parse_sexpr(input).expect_err("rogue `)` after a 2-byte UTF-8 char must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset, 2,
                "byte offset must be 2 (true UTF-8 byte position), NOT 1 (char ordinal); \
                 this guards against a future contributor reverting to chars().enumerate()"
            );
            assert!(
                message.contains("unexpected closing paren"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn imports_model_offset_and_rotation() {
    let s = parse_sexpr(
        r#"(model "m.step" (offset (xyz 3.0 2.5643 0)) (scale (xyz 1 1 1)) (rotate (xyz 0 0 90)))"#,
    )
    .unwrap();
    assert_eq!(xyz_child(&s, "offset"), Some((3.0, 2.5643, 0.0)));
    assert_eq!(xyz_child(&s, "rotate"), Some((0.0, 0.0, 90.0)));
}

#[test]
fn imported_model_offset_is_recentered_with_pads() {
    let p =
        format!("{DOCS}/430450600/KiCADv6/footprints.pretty/CONN_SD-43045-001_06_MOL.kicad_mod");
    let fp = import_kicad_mod(&p, Role::Connector, &[]).unwrap();

    let pad1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (pad1.offset.x.to_mm() - 3.0).abs() < 1e-4 && (pad1.offset.y.to_mm() + 2.5643).abs() < 1e-4,
        "pad 1 is translated into the courtyard-centred frame"
    );
    let (_, offset, _, _) = fp
        .model
        .as_ref()
        .expect("Molex 0430450600 footprint carries its STEP model");
    assert!(
        (offset.0 - 6.0).abs() < 1e-4 && offset.1.abs() < 1e-4,
        "model offset is translated by the same courtyard-centre shift as the pads"
    );
}

#[test]
fn imports_real_iso7740_footprint() {
    let p = format!("{DOCS}/ISO7740DBQR/KiCADv6/footprints.pretty/DBQ0016A_M.kicad_mod");
    let fp = import_kicad_mod(&p, Role::ActiveIc, &["8", "16"]).unwrap();
    assert_eq!(fp.pads.len(), 16, "exact pad count from the vendor file");
    assert_eq!(fp.pad_index("1"), Some(0));
    assert!(fp.pad_index("16").is_some());
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!((p1.offset.x.to_mm() + 2.825).abs() < 1e-3);
    assert!((p1.offset.y.to_mm() + 2.2225).abs() < 1e-3);
    assert!(
        (p1.size.0.to_mm() - 1.65).abs() < 1e-2,
        "rotated long axis (1.65 mm) must lie along X (outward), got {:.3}",
        p1.size.0.to_mm()
    );
    assert!(
        (p1.size.1.to_mm() - 0.4).abs() < 1e-2,
        "rotated short axis (0.4 mm) must lie along the pin pitch (Y), got {:.3}",
        p1.size.1.to_mm()
    );
    let p2 = &fp.pads[fp.pad_index("2").unwrap()];
    let pitch = (p1.offset.y.to_mm() - p2.offset.y.to_mm()).abs();
    assert!(
        p1.size.1.to_mm() < pitch,
        "pad Y-extent {:.3} mm must be below the {:.3} mm pitch (no overlap)",
        p1.size.1.to_mm(),
        pitch
    );
    assert!(fp.pads[fp.pad_index("16").unwrap()].power_pin);
    assert!(fp.courtyard.0.to_mm() > 8.0 && fp.courtyard.1.to_mm() > 5.5);
}

#[test]
fn imports_real_hv7355_qfn56() {
    let p = format!("{DOCS}/HV7355K6_G/KiCADv6/footprints.pretty/QFN56_8X8MC_MCH.kicad_mod");
    let fp = import_kicad_mod(&p, Role::ActiveIc, &[]).unwrap();
    assert_eq!(fp.pads.len(), 57, "real QFN56 pad map");
    assert_eq!(fp.pad_names.len(), fp.pads.len());
}

#[test]
fn imports_real_molex_transducer_header_with_board_locks() {
    let p = format!("{DOCS}/430452400/MOLEX_430452400.kicad_mod");
    let fp = import_kicad_mod(&p, Role::Connector, &[]).unwrap();
    assert_eq!(
        fp.pads.len(),
        26,
        "24 signal pins plus two NPTH board-lock holes"
    );
    assert_eq!(fp.pad_index("1"), Some(0));
    assert_eq!(fp.pad_index("24"), Some(23));
    assert!((fp.courtyard.0.to_mm() - 42.69).abs() < 0.01);
    assert!((fp.courtyard.1.to_mm() - 14.78).abs() < 0.01);
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (p1.offset.x.to_mm() - 16.5).abs() < 0.01 && (p1.offset.y.to_mm() + 2.8).abs() < 0.01,
        "pad coordinates are centred to the courtyard origin"
    );
    let board_locks = fp.pad_names.iter().filter(|name| name.is_empty()).count();
    assert_eq!(
        board_locks, 2,
        "NPTH holes are retained as mechanical (empty-designator) keepouts"
    );
    assert!(
        fp.pads.len() - board_locks == 24,
        "exactly 24 electrical pins remain"
    );
}

#[test]
fn imports_real_molex_430450400_hv_power_header() {
    let p = format!(
        "{DOCS}/430450400/KiCADv6/footprints.pretty/Molex_Micro-Fit_3.0_43045-0400_2x02_P3.00mm_Horizontal.kicad_mod"
    );
    let fp = import_kicad_mod(&p, Role::Connector, &["1", "2", "3", "4"]).unwrap();
    assert_eq!(fp.pads.len(), 5, "four pins plus one NPTH board lock");
    assert_eq!(fp.pad_index("1"), Some(1));
    assert_eq!(fp.pad_index("4"), Some(4));
    let board_locks = fp.pad_names.iter().filter(|name| name.is_empty()).count();
    assert_eq!(
        board_locks, 1,
        "NPTH board lock is retained as a mechanical keepout"
    );
    assert_eq!(
        fp.pads.len() - board_locks,
        4,
        "exactly four electrical power pins remain"
    );
    assert!((fp.courtyard.0.to_mm() - 11.16).abs() < 0.01);
    assert!((fp.courtyard.1.to_mm() - 13.67).abs() < 0.01);
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (p1.offset.x.to_mm() + 1.5).abs() < 0.01 && (p1.offset.y.to_mm() - 2.585).abs() < 0.01,
        "pad 1 is centred to the courtyard coordinate frame"
    );
    assert!(
        ["1", "2", "3", "4"]
            .iter()
            .all(|pin| fp.pads[fp.pad_index(pin).unwrap()].power_pin),
        "all four HV input pins are classified as power-carrying pads"
    );
    let (path, offset, rotate, envelope) = fp
        .model
        .as_ref()
        .expect("Molex 0430450400 footprint carries its STEP model token");
    assert_eq!(path, "${KICAD10_3DMODEL_DIR}/Connector_Molex.3dshapes/Molex_Micro-Fit_3.0_43045-0400_2x02_P3.00mm_Horizontal.step");
    assert!(
        (offset.0 + 1.5).abs() < 1e-6 && (offset.1 - 2.585).abs() < 1e-6 && offset.2 == 0.0,
        "model offset follows the courtyard-centred footprint frame"
    );
    assert_eq!(*rotate, (0.0, 0.0, 0.0));
    assert_eq!(*envelope, None);
}

#[test]
fn imports_real_molex_power_header_model_transform() {
    let p =
        format!("{DOCS}/430450600/KiCADv6/footprints.pretty/CONN_SD-43045-001_06_MOL.kicad_mod");
    let fp = import_kicad_mod(&p, Role::Connector, &[]).unwrap();
    assert_eq!(fp.pads.len(), 7, "six pins plus one NPTH board lock");
    assert_eq!(fp.pad_index("1"), Some(0));
    let p1 = &fp.pads[fp.pad_index("1").unwrap()];
    assert!(
        (p1.offset.x.to_mm() - 3.0).abs() < 1e-4 && (p1.offset.y.to_mm() + 2.5643).abs() < 1e-4,
        "pad 1 is centred to the courtyard coordinate frame"
    );
    let (path, offset, rotate, envelope) = fp
        .model
        .as_ref()
        .expect("Molex 0430450600 footprint carries its STEP model");
    assert_eq!(path, "docs/cad_models/430450600_stp/430450600.stp");
    assert!(
        (offset.0 - 6.0).abs() < 1e-4 && offset.1.abs() < 1e-4 && offset.2 == 0.0,
        "model offset follows the courtyard-centred footprint frame"
    );
    assert_eq!(*rotate, (0.0, 0.0, 0.0));
    assert_eq!(*envelope, None);
}

// ────────────────────────────────────────────────────────────────────────────
// Section D — Tests lifted from src/place/component.rs::tests
// (placement, Rect overlap, assembly clearance). The `lib()` helper lives at
// the top of `tests.rs` (lifted from the inline `mod tests` block).
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn courtyard_follows_rotation() {
    let l = lib();
    let mut c = Component {
        fp: 0,
        nets: vec![None],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let r0 = c.courtyard(&l);
    assert_eq!(r0.max.x - r0.min.x, Nm::from_mm(8.0));
    c.placement.rot = Rot::R90;
    let r90 = c.courtyard(&l);
    assert_eq!(r90.max.x - r90.min.x, Nm::from_mm(4.0)); // axes swapped
}

#[test]
fn pad_position_rotates_about_centre() {
    let l = lib();
    let c = Component {
        fp: 0,
        nets: vec![None],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R90,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    // R90 maps the (+3,0) pad to (0,+3): absolute (10, 13).
    assert_eq!(
        c.pad_pos(&l, 0),
        Point::new(Nm::from_mm(10.0), Nm::from_mm(13.0))
    );
}

#[test]
fn overlap_area_zero_when_disjoint() {
    let a = Rect {
        min: Point::new(Nm(0), Nm(0)),
        max: Point::new(Nm(10), Nm(10)),
    };
    let b = Rect {
        min: Point::new(Nm(20), Nm(0)),
        max: Point::new(Nm(30), Nm(10)),
    };
    assert_eq!(a.overlap_area(b), 0.0);
    assert!(a.overlap_area(a) > 0.0);
}

#[test]
fn component_clearance_detects_inflated_courtyard_overlap() {
    use crate::place::component::component_clearance_violations;
    let l = lib();
    let mk = |refdes: &str, x: f64| Component {
        fp: 0,
        nets: vec![None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let touching = vec![mk("U1", 10.0), mk("U2", 19.0)];
    assert!(
        component_clearance_violations(&touching, &l, Nm::from_mm(1.0)).is_empty(),
        "1.0 mm physical gap exactly holds a 1.0 mm courtyard clearance"
    );
    let too_close = vec![mk("U1", 10.0), mk("U2", 18.2)];
    let v = component_clearance_violations(&too_close, &l, Nm::from_mm(1.0));
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].first, "U1");
    assert_eq!(v[0].second, "U2");
    assert!(v[0].overlap_mm2 > 0.0);
}

// ────────────────────────────────────────────────────────────────────────────
// Section E — Tests lifted from src/place/symbol_import.rs::tests.
// The `PinMap` round-trip covers the function-name ↔ pad-number identity the
// schematic gives us; the two byte-tracking pinning tests lock the Phase 1d
// polish contract for unclosed quoted tokens (name + number). Together they
// exhaustively test the module's parse + lookup surface without touching the
// real vendor .kicad_sym fixtures (whose exact pinmap shape is datasheet-
// specific).
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn pinmap_name_and_number_round_trip_lookup() {
    // Synthesise the simplified parse output: pairs are in file order; the function
    // name is the first column, the pad number is the second. Round-trip every pair
    // through `number_of(name)` and `name_of(number)` and assert the missing
    // lookups are absent (NOT just empty), so a contributor swapping an
    // implementation for `unwrap_or_default()` is caught.
    let pm = PinMap {
        pins: vec![
            ("VPP".to_string(), "1".to_string()),
            ("GND".to_string(), "2".to_string()),
            ("IN".to_string(), "3".to_string()),
        ],
    };
    assert_eq!(pm.number_of("VPP"), Some("1"));
    assert_eq!(pm.number_of("GND"), Some("2"));
    assert_eq!(pm.number_of("IN"), Some("3"));
    assert_eq!(pm.name_of("1"), Some("VPP"));
    assert_eq!(pm.name_of("2"), Some("GND"));
    assert_eq!(pm.name_of("3"), Some("IN"));
    assert_eq!(pm.number_of("NOPE"), None);
    assert_eq!(pm.name_of("99"), None);
    assert_eq!(pm.len(), 3);
    assert!(!pm.is_empty());
}

#[test]
fn pinmap_numbers_of_includes_all_pads_with_same_name() {
    // A multi-pin function name (many GND pads on a large package) is the use case
    // `numbers_of` exists for. The parser preserves file order, so the returned
    // designators must match the order in the source symbol. A single-pin function
    // name returns one designator; a missing function name returns an empty vec
    // (NOT None, as the missing-as-name case for `number_of` does).
    let pm = PinMap {
        pins: vec![
            ("GND".to_string(), "5".to_string()),
            ("VPP".to_string(), "1".to_string()),
            ("GND".to_string(), "6".to_string()),
            ("GND".to_string(), "7".to_string()),
        ],
    };
    assert_eq!(pm.numbers_of("GND"), vec!["5", "6", "7"]);
    assert_eq!(pm.numbers_of("VPP"), vec!["1"]);
    assert!(pm.numbers_of("NOPE").is_empty());
}

#[test]
fn pinmap_empty_map_reports_empty_state() {
    // `PinMap::default()` is the empty-state constructor. Every lookup must drop
    // through: `len() == 0`, `is_empty()`, `number_of(None)`, `numbers_of([])`.
    // A contributor wrapping these in `Option` would silently change the API.
    let pm = PinMap::default();
    assert_eq!(pm.len(), 0);
    assert!(pm.is_empty());
    assert!(pm.number_of("ANY").is_none());
    assert!(pm.numbers_of("ANY").is_empty());
}

#[test]
fn unclosed_quoted_name_token_reports_byte_offset_of_open_quote() {
    // The trailing `)` would close the `(pin …)` form with a `Vec<(usize, bool,
    // String)>` that contains zero events, so the caller falls through to the
    // `no_pins` early-return — which would mask the real bug. The Phase 1d polish
    // surfaces `Manifest::Parse` from inside `quoted_events` instead. The byte
    // offset carried in the envelope is `qstart - 1`, which lands on the opening
    // `"` of the unclosed name token (the byte just before `qstart = from + idx +
    // plen`).
    use std::io::Write;
    let dir = "target/tmp";
    std::fs::create_dir_all(dir).unwrap_or_else(|_| panic!("create {dir}"));
    let path = format!("{dir}/phase_2c_unclosed_name.kicad_sym");
    // 11 bytes precede the opening `"` of the unclosed name: `(pin (name "`
    let mut f = std::fs::File::create(&path).expect("create symbol file");
    f.write_all(b"(pin (name \"missing-end)").unwrap();
    let err = import_symbol_pinmap(&path).expect_err("unclosed quoted name must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset, 11,
                "byte offset must point at the opening `\"` of the unclosed name token (got {offset})"
            );
            assert!(
                message.contains("unclosed quoted token"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("unclosed quoted name token must produce Manifest::Parse; got {err:?}"),
    }
}

#[test]
fn unclosed_quoted_number_token_reports_byte_offset_of_open_quote() {
    // A valid name `"A"` closes cleanly so the parser keeps going; the next
    // `(number "B` opens the byte but never finds the closing `"` before EOF.
    // The envelope must point at the OPENING `"` of the broken number token, NOT
    // at the closed name token above it — the byte-tracking pin prevents a
    // future contributor from accidentally returning the last successfully-parsed
    // position when reporting the failure.
    use std::io::Write;
    let dir = "target/tmp";
    std::fs::create_dir_all(dir).unwrap_or_else(|_| panic!("create {dir}"));
    let path = format!("{dir}/phase_2c_unclosed_number.kicad_sym");
    // 24 bytes precede the opening `"` of the unclosed number:
    // `(pin (name "A") (number "`
    let mut f = std::fs::File::create(&path).expect("create symbol file");
    f.write_all(b"(pin (name \"A\") (number \"B)").unwrap();
    let err = import_symbol_pinmap(&path).expect_err("unclosed quoted number must fail");
    match &err {
        crate::error::Error::Manifest(crate::error::Manifest::Parse { offset, message }) => {
            assert_eq!(
                *offset, 24,
                "byte offset must point at the opening `\"` of the unclosed number token (got {offset})"
            );
            assert!(
                message.contains("unclosed quoted token"),
                "message names the failure mode: {message}"
            );
        }
        _ => panic!("unclosed quoted number token must produce Manifest::Parse; got {err:?}"),
    }
}
