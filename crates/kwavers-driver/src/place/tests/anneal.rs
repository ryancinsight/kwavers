//! Annealing integration tests — moved verbatim from `place/tests/energy.rs`.
//!
//! These tests run the full SA loop and assert placement quality outcomes.
//! Energy unit tests (asserting specific energy term values) remain in `energy.rs`.

use super::*;

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

    anneal_fn(&mut comps, &lib, &cfg, &[0], &params, None);
    anneal_fn(&mut comps, &lib, &cfg, &[1], &params, None);

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
    let after = anneal_fn(&mut comps, &lib, &cfg, &[1], &params, None);
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
    let after = anneal_fn(&mut comps, &lib, &cfg, &movable, &params, None);
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
    anneal_fn(&mut comps, &lib, &cfg, &[0], &params, Some(&cg));
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
        anneal_fn(&mut c, &lib, &cfg, &[0, 1], &params, None);
        (c[0].placement, c[1].placement)
    };
    assert_eq!(run(), run(), "same seed must give identical placement");
}
