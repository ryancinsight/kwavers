//! Tests for the `pipeline` slice (Phase 4m carve-out), verbatim from the flat `mod tests`.

use super::config::role_dissipation_w;
use super::cooptimize::grid_occupancy_shorts;
use super::*;
use crate::board::{Board, NetId};
use crate::board::{LayerId, NetClassKind};
use crate::cost::PhysicsCost;
use crate::geom::Nm;
use crate::geom::{GridSpec, Point};
use crate::place::component::Component;
use crate::place::component::Placement;
use crate::place::footprint::FootprintDef;
use crate::place::footprint::{PadDef, Role};
use crate::place::rotation::Rot;
use crate::route::grid::Grid;
use crate::route::grid::NodeId;
use crate::route::Router;
use crate::rules::CreepageRule;
use crate::rules::DesignRules;
use std::collections::HashSet;
fn empty_result(report: crate::audit::FaultReport, complete: bool, legal: bool) -> CoOptResult {
    CoOptResult {
        board: Board::new(
            GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(1.0), 2).unwrap(),
        ),
        comps: Vec::new(),
        report,
        legal,
        complete,
        rounds_run: 1,
        layer_count: 2,
        layers_used: 0,
    }
}

#[test]
fn manufacturing_clean_requires_internal_drc_lvs_and_route_status() {
    let cfg = CoOpt::default();
    let lib = Vec::new();
    let clean = empty_result(crate::audit::FaultReport::default(), true, true);
    assert!(clean.manufacturing_clean(&lib, &cfg));
    assert_eq!(clean.manufacturing_blockers(&lib, &cfg), Vec::<&str>::new());

    let incomplete = empty_result(crate::audit::FaultReport::default(), false, true);
    assert!(!incomplete.manufacturing_clean(&lib, &cfg));
    assert_eq!(
        incomplete.manufacturing_blockers(&lib, &cfg),
        vec!["incomplete LVS connectivity"]
    );

    let illegal = empty_result(crate::audit::FaultReport::default(), true, false);
    assert!(!illegal.manufacturing_clean(&lib, &cfg));
    assert_eq!(
        illegal.manufacturing_blockers(&lib, &cfg),
        vec!["illegal routed capacity"]
    );

    let dirty_report = crate::audit::FaultReport {
        clearance_violations: 1,
        ..Default::default()
    };
    let dirty = empty_result(dirty_report, true, true);
    assert!(!dirty.manufacturing_clean(&lib, &cfg));
    assert_eq!(
        dirty.manufacturing_blockers(&lib, &cfg),
        vec!["hard internal DRC violations"]
    );
}

#[test]
fn bga_balls_fan_out_via_in_pad_to_an_inner_layer() {
    // A 3×3 BGA at 1 mm pitch with two nets on two balls. `place_to_board` must drop each used
    // ball via-in-pad to the first inner layer (a fanout via at the ball, terminal on layer 1) —
    // so the buried balls become routable off the congested top layer.
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let a = board.add_net("A", NetClassKind::Signal);
    let b = board.add_net("B", NetClassKind::Signal);
    let lib = vec![FootprintDef::bga("U", 3, 3, Nm::from_mm(1.0), &[])];
    // Net the centre ball (index 4) and a corner ball (index 0); rest unconnected.
    let mut nets = vec![None; 9];
    nets[4] = Some(a);
    nets[0] = Some(b);
    let comps = vec![Component {
        fp: 0,
        nets,
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    // Standard (through-hole) stackup: each used ball drops a *filled through-hole* (VIPPO)
    // via-in-pad, and its terminal escapes to the inner layer.
    let inputs = place_to_board(&mut board, &comps, &lib, &DesignRules::holohv());
    assert_eq!(board.vias.len(), 2, "a via-in-pad fanout per used ball");
    assert!(
        board.vias.iter().all(|v| v.from == LayerId(0)
            && v.to == LayerId(3)
            && v.kind == crate::board::ViaKind::Through
            && v.filled),
        "standard stackup ⇒ filled through-hole (VIPPO) in pad"
    );
    for t in &inputs.terminals {
        for group in &t.terminal_groups {
            assert_eq!(group.len(), 1, "BGA access is the escaped inner node");
            let (_, _, layer) = spec.node_coords(group[0].0);
            assert_eq!(layer, 1, "BGA terminal escaped to the inner layer");
        }
    }

    // HDI stackup: the same escape is a laser **micro-via in pad, plated over** (F.Cu→In1).
    let mut board2 = Board::new(spec);
    let a2 = board2.add_net("A", NetClassKind::Signal);
    let mut nets2 = vec![None; 9];
    nets2[4] = Some(a2);
    let comps2 = vec![Component {
        fp: 0,
        nets: nets2,
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    let hdi = DesignRules {
        via_policy: crate::rules::ViaPolicy::Hdi,
        ..DesignRules::holohv()
    };
    place_to_board(&mut board2, &comps2, &lib, &hdi);
    let v = &board2.vias[0];
    assert!(
        v.from == LayerId(0)
            && v.to == LayerId(1)
            && v.kind == crate::board::ViaKind::Micro
            && v.filled,
        "HDI stackup ⇒ laser micro-via-in-pad (VIPPO)"
    );
}

#[test]
fn routes_respect_foreign_pad_halos() {
    // Net A spans left→right at y=4; net B's pad sits dead centre on that line, so A's straight
    // path would graze B's pad. With the halo, A must detour around it (and still connect).
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(8.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
    let a = board.add_net("A", NetClassKind::Hv);
    let b = board.add_net("B", NetClassKind::Signal);
    let fp = |role| {
        FootprintDef::new(
            "P",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            role,
            vec![PadDef {
                offset: Point::new(Nm(0), Nm(0)),
                size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
                layers: vec![LayerId(0)],
                power_pin: false,
            }],
        )
    };
    let lib = vec![fp(Role::ActiveIc), fp(Role::Connector)];
    let comp = |fp, refdes: &str, net, x| Component {
        fp,
        nets: vec![Some(net)],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(4.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![
        comp(0, "A1", a, 2.0),
        comp(0, "A2", a, 18.0),
        comp(1, "B1", b, 10.0),
    ];

    let inputs = place_to_board(&mut board, &comps, &lib, &DesignRules::holohv());
    let b_halo: HashSet<NodeId> = inputs
        .obstacles
        .iter()
        .filter(|o| o.net == Some(b))
        .flat_map(|o| o.nodes_hv.iter().copied())
        .collect();
    assert!(
        !b_halo.is_empty(),
        "B's pad must inflate to a non-empty halo"
    );

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        1.0,
        1.0,
    );
    let mut router = Router::new(Grid::new(spec), cost);
    let outcome = router.route_with_obstacles(&inputs.terminals, &inputs.obstacles);

    let ai = inputs.terminals.iter().position(|t| t.net == a).unwrap();
    assert!(outcome.complete, "A must connect by routing around B");
    let a_nodes: HashSet<NodeId> = outcome.routes[ai].nodes.iter().copied().collect();
    assert!(
        a_nodes.is_disjoint(&b_halo),
        "net A's route must avoid every node of B's pad clearance halo"
    );
}

#[test]
fn drilled_pads_route_from_any_copper_layer() {
    // A drilled connector pin is a plated barrel. It is one logical terminal with access on
    // every copper layer, so an automated route can enter the pin from an inner layer without
    // creating a separate same-location via.
    let spec = GridSpec::cover(Nm::from_mm(12.0), Nm::from_mm(6.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let net = board.add_net("TX", NetClassKind::Hv);
    let smd = FootprintDef::new(
        "SMD",
        (Nm::from_mm(1.0), Nm::from_mm(1.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.7), Nm::from_mm(0.7)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    );
    let pth = FootprintDef::new(
        "PTH",
        (Nm::from_mm(1.5), Nm::from_mm(1.5)),
        Role::Connector,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            layers: (0..spec.nlayers)
                .map(|layer| LayerId(layer as u16))
                .collect(),
            power_pin: false,
        }],
    );
    let lib = vec![smd, pth];
    let comps = vec![
        Component {
            fp: 0,
            nets: vec![Some(net)],
            refdes: "U1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(3.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        },
        Component {
            fp: 1,
            nets: vec![Some(net)],
            refdes: "J1".into(),
            placement: Placement {
                pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(3.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            locked: false,
            ..Default::default()
        },
    ];

    let inputs = place_to_board(&mut board, &comps, &lib, &DesignRules::holohv());
    let groups = &inputs.terminals[0].terminal_groups;
    assert_eq!(groups.len(), 2, "one logical terminal per pad");
    assert!(
        groups.iter().any(|group| group.len() == spec.nlayers),
        "the drilled pad must expose every copper layer as equivalent access"
    );

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        1.0,
        1.0,
    );
    let mut router = Router::new(Grid::new(spec), cost);
    let outcome = router.route_with_obstacles(&inputs.terminals, &inputs.obstacles);

    assert!(
        outcome.complete,
        "the SMD pad must connect to the drilled pin"
    );
    assert!(
        outcome.legal,
        "drilled-pad access must not overuse routing resources"
    );
    router.apply_to_board(
        &mut board,
        &inputs.terminals,
        &outcome,
        &DesignRules::holohv(),
    );
    assert!(
        board.tracks.iter().any(|track| track.net == net),
        "a complete drilled-pad route must emit copper for the connected net"
    );
}

#[test]
fn routes_avoid_unconnected_pad_copper() {
    // A real footprint can carry pads with no net. They are still physical copper: a route that
    // crossed one would short to dead copper. The obstacle model must block an unconnected pad
    // (net = None) for *every* net, so net A detours around it.
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(8.0), Nm::from_mm(0.5), 2).unwrap();
    let mut board = Board::new(spec);
    let a = board.add_net("A", NetClassKind::Signal);
    // A two-pad part: pad 0 carries net A, pad 1 carries **no net** (unconnected copper) and sits
    // on the straight path between the two A terminals.
    let fp = FootprintDef::new(
        "P",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    );
    let lib = vec![fp];
    let comp = |refdes: &str, net, x| Component {
        fp: 0,
        nets: vec![net],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(4.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![
        comp("A1", Some(a), 2.0),
        comp("NC", None, 10.0), // unconnected pad on the path
        comp("A2", Some(a), 18.0),
    ];

    let inputs = place_to_board(&mut board, &comps, &lib, &DesignRules::holohv());
    // The unconnected pad produced an obstacle (net = None) — the model no longer skips it.
    let nc_halo: HashSet<NodeId> = inputs
        .obstacles
        .iter()
        .filter(|o| o.net.is_none())
        .flat_map(|o| o.nodes_signal.iter().copied())
        .collect();
    assert!(
        !nc_halo.is_empty(),
        "the unconnected pad must inflate a keepout"
    );

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        1.0,
        1.0,
    );
    let mut router = Router::new(Grid::new(spec), cost);
    let outcome = router.route_with_obstacles(&inputs.terminals, &inputs.obstacles);
    let ai = inputs.terminals.iter().position(|t| t.net == a).unwrap();
    assert!(
        outcome.complete,
        "A must connect by routing around the dead copper"
    );
    let a_nodes: HashSet<NodeId> = outcome.routes[ai].nodes.iter().copied().collect();
    assert!(
        a_nodes.is_disjoint(&nc_halo),
        "net A's route must avoid the unconnected pad's copper keepout"
    );
}

#[test]
fn cooptimize_produces_short_free_board() {
    // A congested HV/LV mix on a small board: several nets whose flight lines cross, forcing the
    // router to use vias and detours. The co-optimiser must still return a complete, legal,
    // geometrically short-free board — the property the via-keepout / pad-halo machinery exists
    // to guarantee.
    let spec = GridSpec::cover(Nm::from_mm(40.0), Nm::from_mm(28.0), Nm::from_mm(0.5), 4).unwrap();
    let mut tmpl = Board::new(spec);
    let nets: Vec<NetId> = (0..4)
        .map(|i| {
            let class = if i == 0 {
                NetClassKind::Hv
            } else {
                NetClassKind::Signal
            };
            tmpl.add_net(format!("N{i}"), class)
        })
        .collect();

    let fp = FootprintDef::new(
        "Q",
        (Nm::from_mm(3.0), Nm::from_mm(3.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(-Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let lib = vec![fp];
    // Eight 2-pad parts wiring the four nets across the board so flight lines interleave.
    let mk = |a: NetId, b: NetId, x, y| Component {
        fp: 0,
        nets: vec![Some(a), Some(b)],
        refdes: "U".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![
        mk(nets[0], nets[1], 7.0, 7.0),
        mk(nets[1], nets[2], 33.0, 7.0),
        mk(nets[2], nets[3], 7.0, 21.0),
        mk(nets[3], nets[0], 33.0, 21.0),
        mk(nets[0], nets[2], 20.0, 7.0),
        mk(nets[1], nets[3], 20.0, 21.0),
    ];

    let cfg = CoOpt {
        rounds: 3,
        patience: 2,
        place: crate::place::PlaceConfig {
            board: (Nm::from_mm(40.0), Nm::from_mm(28.0)),
            margin: Nm::from_mm(1.0),
            thermal_spacing: Nm::from_mm(6.0),
            courtyard_clearance: Nm::from_mm(0.5),
            weights: crate::place::PlaceWeights::default(),
            isolation_axis: crate::place::Axis::X,
        },
        anneal: crate::place::AnnealParams {
            steps: 3_000,
            ..Default::default()
        },
        emi_weight: 0.0, // this test isolates short-freeness, not EMI separation
        ..Default::default()
    };
    let r = cooptimize(&tmpl, comps, &lib, &DesignRules::holohv(), &cfg);

    assert!(r.complete, "every net must connect");
    assert!(r.legal, "routing must hold capacity (no over-use)");
    assert_eq!(
        r.report.via_adjacency, 0,
        "no different-net vias within annular-ring clearance"
    );
    assert_eq!(r.report.dangling, 0, "no antenna / open-fault track ends");
    assert_eq!(
        grid_occupancy_shorts(&r.board),
        0,
        "no different-net copper sharing a cell/layer or via column"
    );
}

#[test]
fn thermal_feedback_flattens_the_temperature_field() {
    // Three dissipative ICs, each on its OWN self-contained 2-pad net so routing is a trivial
    // internal stub regardless of placement — routing risk is 0 for every layout, isolating the
    // thermal objective. Enabling the thermal guidance (`thermal_weight > 0`) must reach a
    // *strictly lower* peak steady-state temperature than the thermal-blind ablation: the
    // physics must measurably flatten the field, not merely be wired in.
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(40.0), Nm::from_mm(0.5), 4).unwrap();
    let mut tmpl = Board::new(spec);
    let nets: Vec<NetId> = (0..3)
        .map(|i| tmpl.add_net(format!("P{i}"), NetClassKind::Signal))
        .collect();

    // A hot IC with two pads on the *same* net: its route never leaves the part.
    let ic = FootprintDef::new(
        "HOT",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(-Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let lib = vec![ic];
    let mk = |net: NetId, x, y| Component {
        fp: 0,
        nets: vec![Some(net), Some(net)],
        refdes: "U".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    // Start them clustered in the centre so there is headroom to spread.
    let comps = vec![
        mk(nets[0], 28.0, 20.0),
        mk(nets[1], 31.0, 20.0),
        mk(nets[2], 34.0, 20.0),
    ];

    // Small placer thermal-spacing so the *guidance* (feedback + judge), not the placer's own
    // pairwise term, is what drives the spread under test.
    let base = crate::place::PlaceConfig {
        board: (Nm::from_mm(60.0), Nm::from_mm(40.0)),
        margin: Nm::from_mm(2.0),
        thermal_spacing: Nm::from_mm(1.0),
        courtyard_clearance: Nm::from_mm(1.0),
        weights: crate::place::PlaceWeights::default(),
        isolation_axis: crate::place::Axis::X,
    };
    let anneal = crate::place::AnnealParams {
        steps: 4_000,
        ..Default::default()
    };
    let run = |tw: f64| -> CoOptResult {
        let cfg = CoOpt {
            rounds: 5,
            patience: 5,
            place: base,
            anneal,
            thermal_weight: tw,
            seed_groups: false, // start clustered so thermal guidance must do the spreading
            ..Default::default()
        };
        cooptimize(&tmpl, comps.clone(), &lib, &DesignRules::holohv(), &cfg)
    };
    let peak = |r: &CoOptResult| -> f64 {
        crate::physics::thermal::solve_board(
            spec,
            &r.comps,
            &lib,
            |fp| role_dissipation_w(fp.role),
            20.0,
            1.6e-3,
            10.0,
            150,
        )
        .peak()
    };

    let off = run(0.0);
    let on = run(8.0);
    // Sanity: the trivial isolated nets route on both (no confounding risk differences).
    assert!(off.complete && on.complete, "both placements must route");
    let (peak_off, peak_on) = (peak(&off), peak(&on));
    assert!(
        peak_on < peak_off,
        "thermal guidance must lower the peak temperature: on={peak_on:.4} off={peak_off:.4}"
    );
}

#[test]
fn density_feedback_spreads_the_whole_bom() {
    // Five identical PASSIVE parts — so the active-IC `thermal`/`ic_spread` terms never act on them
    // — each on its OWN self-contained 2-pad net (routing is a trivial internal stub for any layout,
    // isolating the density objective). Enabling the density guidance (`density_weight > 0`) must
    // reach a *strictly lower* peak component-area density than the density-blind ablation: the
    // electrostatic area-as-charge spreading must measurably flatten the density field, not merely
    // be wired in. `thermal_spacing = 0` removes the hard pairwise pad, so the physics alone spreads.
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(40.0), Nm::from_mm(0.5), 4).unwrap();
    let mut tmpl = Board::new(spec);
    let nets: Vec<NetId> = (0..5)
        .map(|i| tmpl.add_net(format!("N{i}"), NetClassKind::Signal))
        .collect();
    let part = FootprintDef::new(
        "PASV",
        (Nm::from_mm(4.0), Nm::from_mm(4.0)),
        Role::Passive,
        vec![
            PadDef {
                offset: Point::new(-Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let lib = vec![part];
    let mk = |net: NetId, x, y| Component {
        fp: 0,
        nets: vec![Some(net), Some(net)],
        refdes: "R".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    // Clustered in the centre so there is headroom to spread.
    let comps = vec![
        mk(nets[0], 28.0, 20.0),
        mk(nets[1], 30.0, 20.0),
        mk(nets[2], 32.0, 20.0),
        mk(nets[3], 30.0, 18.0),
        mk(nets[4], 30.0, 22.0),
    ];
    let base = crate::place::PlaceConfig {
        board: (Nm::from_mm(60.0), Nm::from_mm(40.0)),
        margin: Nm::from_mm(2.0),
        thermal_spacing: Nm::from_mm(0.0), // no hard padding — the density physics must spread them
        courtyard_clearance: Nm::from_mm(0.5),
        weights: crate::place::PlaceWeights::default(),
        isolation_axis: crate::place::Axis::X,
    };
    let anneal = crate::place::AnnealParams {
        steps: 4_000,
        ..Default::default()
    };
    let run = |dw: f64| -> CoOptResult {
        let cfg = CoOpt {
            rounds: 5,
            patience: 5,
            place: base,
            anneal,
            density_weight: dw,
            seed_groups: false, // start clustered so density guidance must do the spreading
            ..Default::default()
        };
        cooptimize(&tmpl, comps.clone(), &lib, &DesignRules::holohv(), &cfg)
    };
    // Peak of the same area-sourced Poisson field the guidance minimises: lower ⇒ flatter ⇒ spread.
    let density_peak = |r: &CoOptResult| -> f64 {
        crate::physics::thermal::solve_board(
            spec,
            &r.comps,
            &lib,
            |fp| {
                let (w, h) = fp.courtyard;
                w.to_mm() * h.to_mm()
            },
            20.0,
            1.6e-3,
            10.0,
            150,
        )
        .peak()
    };
    let off = run(0.0);
    let on = run(10.0);
    assert!(off.complete && on.complete, "both placements must route");
    let (peak_off, peak_on) = (density_peak(&off), density_peak(&on));
    assert!(
        peak_on < peak_off,
        "density guidance must flatten the area-density field: on={peak_on:.5} off={peak_off:.5}"
    );
}

#[test]
fn emi_guidance_separates_hv_from_lv() {
    // An HV driver and an LV controller, each a 2-pad part on its own net, on a roomy board. EMI
    // guidance must pull the HV and LV pads apart — fewer HV↔LV close pairs than the EMI-blind
    // ablation (and no worse than the as-placed clustered start).
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(40.0), Nm::from_mm(0.5), 4).unwrap();
    let mut tmpl = Board::new(spec);
    let hv = tmpl.add_net("VPP", NetClassKind::Hv);
    let lv = tmpl.add_net("CTRL", NetClassKind::Signal);
    let fp = |role| {
        FootprintDef::new(
            "P",
            (Nm::from_mm(3.0), Nm::from_mm(3.0)),
            role,
            vec![
                PadDef {
                    offset: Point::new(-Nm::from_mm(1.0), Nm(0)),
                    size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
                PadDef {
                    offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                    size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                    layers: vec![LayerId(0)],
                    power_pin: false,
                },
            ],
        )
    };
    let lib = vec![fp(Role::ActiveIc)];
    let mk = |net, x, y| Component {
        fp: 0,
        nets: vec![Some(net), Some(net)],
        refdes: "U".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    // Start them adjacent (HV right next to LV).
    let comps = vec![mk(hv, 28.0, 20.0), mk(lv, 32.0, 20.0)];

    let base = crate::place::PlaceConfig {
        board: (Nm::from_mm(60.0), Nm::from_mm(40.0)),
        margin: Nm::from_mm(2.0),
        thermal_spacing: Nm::from_mm(1.0),
        courtyard_clearance: Nm::from_mm(1.0),
        weights: crate::place::PlaceWeights::default(),
        ..Default::default()
    };
    let run = |ew: f64| {
        let cfg = CoOpt {
            rounds: 5,
            patience: 5,
            place: base,
            anneal: crate::place::AnnealParams {
                steps: 4_000,
                ..Default::default()
            },
            emi_weight: ew,
            ..Default::default()
        };
        let r = cooptimize(&tmpl, comps.clone(), &lib, &DesignRules::holohv(), &cfg);
        let mut rb = tmpl.clone();
        let _ = place_to_board(&mut rb, &r.comps, &lib, &DesignRules::holohv());
        crate::audit::emi_hotspots(&rb, Nm::from_mm(6.0)).len()
    };
    let blind = run(0.0);
    let guided = run(30.0);
    assert!(
        guided <= blind,
        "EMI guidance must not worsen HV↔LV separation: guided={guided} blind={blind}"
    );
    assert_eq!(guided, 0, "strong EMI guidance fully separates HV from LV");
}

#[test]
fn symmetric_seeding_distributes_identical_ics_without_overlap() {
    // Four identical ActiveIc parts all starting at the same board-centre position.
    // seed_symmetric_groups must spread them into a 2×2 grid so cooptimize can route without
    // the first round being immediately dominated by total courtyard overlap.
    // Acceptance: after cooptimize, all four ICs are at distinct positions with no courtyard
    // overlaps (clearance 0.5 mm). Route quality (complete/legal) is not required here
    // because the nets are isolated stubs — the assertion is placement geometry only.
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(40.0), Nm::from_mm(0.5), 4).unwrap();
    let mut tmpl = Board::new(spec);
    let nets: Vec<NetId> = (0..4)
        .map(|i| tmpl.add_net(format!("TX{i}"), NetClassKind::Hv))
        .collect();

    let fp = FootprintDef::new(
        "HV",
        (Nm::from_mm(5.0), Nm::from_mm(5.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(-Nm::from_mm(1.5), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(1.5), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let lib = vec![fp];

    // All four ICs start at the same centre — worst case for the placer.
    let mk = |net: NetId| Component {
        fp: 0,
        nets: vec![Some(net), Some(net)],
        refdes: "U".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(30.0), Nm::from_mm(20.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![mk(nets[0]), mk(nets[1]), mk(nets[2]), mk(nets[3])];

    let cfg = CoOpt {
        rounds: 4,
        patience: 2,
        place: crate::place::PlaceConfig {
            board: (Nm::from_mm(60.0), Nm::from_mm(40.0)),
            margin: Nm::from_mm(2.0),
            thermal_spacing: Nm::from_mm(8.0),
            courtyard_clearance: Nm::from_mm(0.5),
            weights: crate::place::PlaceWeights::default(),
            isolation_axis: crate::place::Axis::X,
        },
        anneal: crate::place::AnnealParams {
            steps: 4_000,
            ..Default::default()
        },
        emi_weight: 0.0,
        ..Default::default()
    };

    let r = cooptimize(&tmpl, comps, &lib, &DesignRules::holohv(), &cfg);

    // Every pair of IC positions must be distinct (no two ICs co-located after spreading).
    for i in 0..r.comps.len() {
        for j in (i + 1)..r.comps.len() {
            let d = r.comps[i].placement.pos.euclid(r.comps[j].placement.pos) * 1.0e-6; // nm → mm
            assert!(
                d > 1.0,
                "ICs {i} and {j} are co-located after seeding+anneal: d={d:.2} mm"
            );
        }
    }
    // No courtyard overlaps (clearance 0.5 mm).
    let violations = crate::place::component_clearance_violations(&r.comps, &lib, Nm::from_mm(0.5));
    assert!(
        violations.is_empty(),
        "no courtyard overlaps after cooptimize with 4 identical ICs: {violations:?}"
    );
}

#[test]
fn cooptimize_min_area_picks_a_small_clean_board() {
    // Two isolated 2-pad parts (each its own net) — trivially routable, so the smallest candidate
    // that fits the placement should win. The minimiser must return a clean board strictly
    // smaller than the largest candidate (it did not just fall back to the biggest).
    let mut tmpl = Board::new(
        GridSpec::cover(Nm::from_mm(40.0), Nm::from_mm(40.0), Nm::from_mm(0.5), 2).unwrap(),
    );
    let a = tmpl.add_net("A", NetClassKind::Signal);
    let b = tmpl.add_net("B", NetClassKind::Signal);
    let fp = FootprintDef::new(
        "Q",
        (Nm::from_mm(3.0), Nm::from_mm(3.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(-Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(1.0), Nm(0)),
                size: (Nm::from_mm(0.8), Nm::from_mm(0.8)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    );
    let lib = vec![fp];
    let comp = |net, x, y| Component {
        fp: 0,
        nets: vec![Some(net), Some(net)],
        refdes: "U".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let comps = vec![comp(a, 10.0, 10.0), comp(b, 30.0, 30.0)];
    let cfg = CoOpt {
        rounds: 2,
        patience: 2,
        anneal: crate::place::AnnealParams {
            steps: 2_000,
            ..Default::default()
        },
        ..Default::default()
    };
    let sizes = [
        (Nm::from_mm(16.0), Nm::from_mm(16.0)),
        (Nm::from_mm(24.0), Nm::from_mm(24.0)),
        (Nm::from_mm(40.0), Nm::from_mm(40.0)),
        (Nm::from_mm(48.0), Nm::from_mm(48.0)),
    ];
    let r = cooptimize_min_area(&tmpl, &comps, &lib, &DesignRules::holohv(), &cfg, &sizes)
        .expect("sizes is non-empty");
    assert!(
        r.complete && r.legal,
        "the minimiser returns a routable board"
    );
    let w_mm = (r.board.spec.nx as i64 - 1) * r.board.spec.pitch.0;
    assert!(
        w_mm < Nm::from_mm(48.0).0,
        "it picked a board smaller than the largest candidate (auto-minimised)"
    );
}
