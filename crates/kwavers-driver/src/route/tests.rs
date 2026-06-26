//! Negotiated-congestion routing tests — PathFinder inner-loop + tree growth + emission checks.
//!
//! Moved out of `src/route/mod.rs` at Phase 2b so the 10 tests below live in their own file
//! (gated `#[cfg(test)]`) per the spec's `route/{mod.rs, grid.rs, search.rs, pathfinder.rs,
//! tests.rs, ...}` layout. The test count is preserved from the pre-move state — the 10 tests
//! below were moved **verbatim** from `src/route/mod.rs::mod tests`, with the `use super::*;`
//! preamble (which previously resolved via `crate::route`'s `pub use` re-exports) now resolving
//! from `crate::route` directly because `src/route/tests.rs` is a sibling sub-module whose
//! `super` is also `crate::route`.

use super::*;
use crate::board::{Board, LayerId, NetClassKind, Pad};
use crate::cost::PhysicsCost;
use crate::geom::{segments_cross, GridSpec, Nm, Point};
use crate::rules::{CreepageRule, DesignRules};
use std::collections::HashSet;

/// Build a terminal node from a board point on a layer.
fn node_at(spec: &GridSpec, x_mm: f64, y_mm: f64, layer: usize) -> NodeId {
    let (ix, iy) = spec.cell_of(Point::new(Nm::from_mm(x_mm), Nm::from_mm(y_mm)));
    NodeId(spec.node_index(ix, iy, layer))
}

/// A flat zero cost so the routing tests isolate the *congestion* mechanism from physics bias.
struct UnitCost;
impl crate::cost::RoutingCost for UnitCost {
    fn node_base(&self, _p: Point, _l: LayerId, _c: NetClassKind) -> f64 {
        1.0
    }
    fn via_cost(&self, _class: NetClassKind) -> f64 {
        4.0
    }
}

#[test]
fn single_net_routes_and_connects() {
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(2.0), Nm::from_mm(0.5), 2).unwrap();
    let grid = Grid::new(spec);
    let mut router = Router::new(grid, UnitCost);
    let nets = vec![NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![
            vec![node_at(&spec, 0.0, 1.0, 0)],
            vec![node_at(&spec, 9.5, 1.0, 0)],
        ],
    }];
    let out = router.route(&nets);
    assert!(out.legal, "single net must be legal");
    assert!(out.complete, "single net must connect");
    assert!(
        !out.routes[0].edges.is_empty(),
        "a non-trivial path must exist"
    );
}

/// Two nets whose straight paths cross at the same cell on one layer. A single-layer router
/// cannot legalise this; negotiated congestion must push one net onto the second layer (a via
/// detour) so that no node is shared. This is the case sequential A* deadlocks on.
#[test]
fn crossing_nets_legalise_via_second_layer() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(8.0), Nm::from_mm(0.5), 2).unwrap();
    let grid = Grid::new(spec);
    let mut router = Router::new(grid, UnitCost);
    // Net A: left->right across the middle row.  Net B: top->bottom across the middle col.
    // They must intersect at the centre cell.
    let a = NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![
            vec![node_at(&spec, 0.0, 4.0, 0)],
            vec![node_at(&spec, 8.0, 4.0, 0)],
        ],
    };
    let b = NetTerminals {
        net: crate::board::NetId(1),
        class: NetClassKind::Signal,
        terminal_groups: vec![
            vec![node_at(&spec, 4.0, 0.0, 0)],
            vec![node_at(&spec, 4.0, 8.0, 0)],
        ],
    };
    let out = router.route(&[a, b]);
    assert!(
        out.legal,
        "crossing nets must legalise (got {} over-used nodes after {} iters)",
        out.overused_nodes, out.iterations
    );
    assert!(out.complete, "both nets must connect");

    // Verify the legality claim independently: no grid node is used by both nets.
    let mut used: HashSet<NodeId> = HashSet::new();
    let mut overlap = false;
    for r in &out.routes {
        let mut this: HashSet<NodeId> = HashSet::new();
        for &n in &r.nodes {
            this.insert(n);
        }
        for n in &this {
            if !used.insert(*n) {
                overlap = true;
            }
        }
    }
    assert!(!overlap, "no node may be shared between the two nets");

    // And at least one net must have taken a via (left the start layer) to deconflict.
    let vias: usize = out
        .routes
        .iter()
        .flat_map(|r| r.edges.iter())
        .filter(|(x, y)| spec.node_coords(x.0).2 != spec.node_coords(y.0).2)
        .count();
    assert!(
        vias > 0,
        "deconfliction requires at least one via to the second layer"
    );
}

#[test]
fn diagonal_routing_avoids_crossed_foreign_diagonal_edges() {
    let spec = GridSpec::cover(Nm::from_mm(3.0), Nm::from_mm(3.0), Nm::from_mm(1.0), 2).unwrap();
    let mut grid = Grid::new(spec);
    grid.set_diagonal_routing(true);
    let mut router = Router::new(grid, UnitCost);
    let n = |ix, iy, layer| NodeId(spec.node_index(ix, iy, layer));
    let a = NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![n(0, 0, 0)], vec![n(1, 1, 0)]],
    };
    let b = NetTerminals {
        net: crate::board::NetId(1),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![n(0, 1, 0)], vec![n(1, 0, 0)]],
    };

    let out = router.route(&[a, b]);
    assert!(out.complete, "both diagonal-corner nets must connect");
    assert!(out.legal, "alternate route must avoid node overuse");

    let mut edges = Vec::new();
    for (route_idx, route) in out.routes.iter().enumerate() {
        for &(u, v) in &route.edges {
            let (ux, uy, ul) = spec.node_coords(u.0);
            let (vx, vy, vl) = spec.node_coords(v.0);
            if ul == vl {
                edges.push((route_idx, ul, spec.point_of(ux, uy), spec.point_of(vx, vy)));
            }
        }
    }
    for (i, a) in edges.iter().enumerate() {
        for b in edges.iter().skip(i + 1) {
            if a.0 != b.0 && a.1 == b.1 {
                assert!(
                    !segments_cross(a.2, a.3, b.2, b.3),
                    "different-net same-layer route edges must not physically cross"
                );
            }
        }
    }
}

#[test]
fn diagonal_routing_avoids_foreign_via_corner_clearance() {
    let spec = GridSpec::cover(Nm::from_mm(3.0), Nm::from_mm(3.0), Nm::from_mm(1.0), 2).unwrap();
    let mut grid = Grid::new(spec);
    grid.set_diagonal_routing(true);
    let mut router = Router::new(grid, UnitCost);
    let n = |ix, iy, layer| NodeId(spec.node_index(ix, iy, layer));
    let via_net = NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![n(0, 1, 0)], vec![n(0, 1, 1)]],
    };
    let diagonal_net = NetTerminals {
        net: crate::board::NetId(1),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![n(0, 0, 0)], vec![n(2, 2, 0)]],
    };

    let out = router.route(&[via_net, diagonal_net]);
    assert!(out.complete, "via and diagonal nets must connect");
    assert!(out.legal, "diagonal net must avoid the foreign via column");
    let diagonal_route = &out.routes[1];
    assert!(
        !diagonal_route.edges.iter().any(|&(u, v)| {
            (u == n(0, 0, 0) && v == n(1, 1, 0)) || (u == n(1, 1, 0) && v == n(0, 0, 0))
        }),
        "the direct diagonal clips the via at the square corner and must be rejected"
    );
}

#[test]
fn diagonal_routing_avoids_foreign_track_corner_clearance() {
    let spec = GridSpec::cover(Nm::from_mm(3.0), Nm::from_mm(3.0), Nm::from_mm(1.0), 2).unwrap();
    let mut grid = Grid::new(spec);
    grid.set_diagonal_routing(true);
    let mut router = Router::new(grid, UnitCost);
    let n = |ix, iy, layer| NodeId(spec.node_index(ix, iy, layer));
    let corner_net = NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![n(1, 0, 0)], vec![n(2, 0, 0)]],
    };
    let diagonal_net = NetTerminals {
        net: crate::board::NetId(1),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![n(0, 0, 0)], vec![n(1, 1, 0)]],
    };

    let out = router.route(&[corner_net, diagonal_net]);

    assert!(out.complete, "corner and diagonal nets must connect");
    assert!(
        out.legal,
        "diagonal net must avoid the foreign track corner"
    );
    let diagonal_route = &out.routes[1];
    assert!(
        !diagonal_route.edges.iter().any(|&(u, v)| {
            (u == n(0, 0, 0) && v == n(1, 1, 0)) || (u == n(1, 1, 0) && v == n(0, 0, 0))
        }),
        "the direct diagonal clips a foreign track corner and must be rejected"
    );
}

#[test]
fn multi_terminal_signal_routes_as_daisy_chain() {
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 1).unwrap();
    let grid = Grid::new(spec);
    let mut router = Router::new(grid, UnitCost);
    let nets = vec![NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![
            vec![node_at(&spec, 5.0, 5.0, 0)],
            vec![node_at(&spec, 2.0, 5.0, 0)],
            vec![node_at(&spec, 8.0, 5.0, 0)],
            vec![node_at(&spec, 5.0, 2.0, 0)],
            vec![node_at(&spec, 5.0, 8.0, 0)],
        ],
    }];
    let out = router.route(&nets);
    assert!(
        out.legal && out.complete,
        "multi-terminal signal must route"
    );

    let mut degree = std::collections::BTreeMap::<NodeId, usize>::new();
    for &(a, b) in &out.routes[0].edges {
        *degree.entry(a).or_default() += 1;
        *degree.entry(b).or_default() += 1;
    }
    let max_degree = degree.values().copied().max().unwrap_or(0);
    assert!(
        max_degree <= 2,
        "high-speed signal routing must form a daisy chain with no T-branch node, got max degree {max_degree}"
    );
}

#[test]
fn high_speed_nets_route_before_power_while_output_order_stays_input_order() {
    let spec = GridSpec::cover(Nm::from_mm(3.0), Nm::from_mm(3.0), Nm::from_mm(1.0), 1).unwrap();
    let grid = Grid::new(spec);
    let params = PathFinderParams {
        max_iter: 1,
        present0: 20.0,
        present_mul: 1.0,
        history_gain: 0.0,
        history_decay: 0.0,
    };
    let mut router = Router::new(grid, UnitCost).with_params(params);
    let center = node_at(&spec, 1.0, 1.0, 0);
    let power = NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Power,
        terminal_groups: vec![
            vec![node_at(&spec, 1.0, 0.0, 0)],
            vec![node_at(&spec, 1.0, 2.0, 0)],
        ],
    };
    let signal = NetTerminals {
        net: crate::board::NetId(1),
        class: NetClassKind::Signal,
        terminal_groups: vec![
            vec![node_at(&spec, 0.0, 1.0, 0)],
            vec![node_at(&spec, 2.0, 1.0, 0)],
        ],
    };

    let out = router.route(&[power, signal]);

    assert_eq!(
        out.routes.len(),
        2,
        "route count must remain aligned with input nets"
    );
    assert!(
        out.routes[1].nodes.contains(&center),
        "the signal net is second in the input, but must reserve the shortest high-speed channel first"
    );
    assert!(
        !out.routes[0].nodes.contains(&center),
        "the power net is first in the input, but must route around the already-reserved high-speed channel"
    );
}

/// End-to-end: physics-guided cost keeps an HV net off the layer/region near an LV pad while
/// still connecting, and the result emits real tracks/vias onto the board.
#[test]
fn physics_cost_routes_and_emits_copper() {
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(6.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let hv = board.add_net("TX0", NetClassKind::Hv);
    let lv = board.add_net("CTRL", NetClassKind::Signal);
    // LV pad sits in the middle; HV must keep creepage from it.
    let lv_pad = Point::new(Nm::from_mm(5.0), Nm::from_mm(3.0));
    board.add_pad(Pad {
        pos: lv_pad,
        layers: vec![LayerId(0)],
        net: Some(lv),
    });
    let hv_a = Point::new(Nm::from_mm(0.0), Nm::from_mm(3.0));
    let hv_b = Point::new(Nm::from_mm(9.5), Nm::from_mm(3.0));
    board.add_pad(Pad {
        pos: hv_a,
        layers: vec![LayerId(0)],
        net: Some(hv),
    });
    board.add_pad(Pad {
        pos: hv_b,
        layers: vec![LayerId(0)],
        net: Some(hv),
    });

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        80.0,
        2.0,
    );
    let grid = Grid::new(spec);
    let mut router = Router::new(grid, cost);
    let nets = vec![NetTerminals {
        net: hv,
        class: NetClassKind::Hv,
        terminal_groups: vec![
            vec![node_at(&spec, 0.0, 3.0, 0)],
            vec![node_at(&spec, 9.5, 3.0, 0)],
        ],
    }];
    let out = router.route(&nets);
    assert!(out.legal && out.complete, "HV net must route");

    // The HV route must avoid the exact LV-pad cell (creepage hazard makes it expensive).
    let lv_cell = node_at(&spec, 5.0, 3.0, 0);
    assert!(
        !out.routes[0].nodes.contains(&lv_cell),
        "physics cost must steer the HV route off the LV-pad cell"
    );

    let rules = crate::rules::DesignRules::holohv();
    router.apply_to_board(&mut board, &nets, &out, &rules);
    assert!(!board.tracks.is_empty(), "routing must emit copper tracks");
    // Emitted HV tracks carry the HV width.
    assert!(board
        .tracks
        .iter()
        .all(|t| t.width == rules.hv_track || t.net != hv));
}

#[test]
fn diagonal_route_takes_fewer_nodes_than_manhattan() {
    // A single net from (0,0) to (4,4) at layer 0 on a 5×5 grid.
    // Manhattan path: 8 moves, 9 nodes. Diagonal path: 4 moves, 5 nodes.
    let spec = GridSpec::cover(Nm::from_mm(4.0), Nm::from_mm(4.0), Nm::from_mm(1.0), 1).unwrap();
    let mut g_diag = Grid::new(spec);
    g_diag.set_diagonal_routing(true);
    let mut router_diag = Router::new(g_diag, UnitCost);
    let a = node_at(&spec, 0.0, 0.0, 0);
    let b = node_at(&spec, 4.0, 4.0, 0);
    let nets = vec![NetTerminals {
        net: crate::board::NetId(0),
        class: NetClassKind::Signal,
        terminal_groups: vec![vec![a], vec![b]],
    }];
    let out_diag = router_diag.route(&nets);
    assert!(out_diag.complete, "diagonal route must connect");
    assert!(out_diag.legal, "diagonal route must be legal");

    let g_axial = Grid::new(spec);
    let mut router_axial = Router::new(g_axial, UnitCost);
    let out_axial = router_axial.route(&nets);

    // Diagonal should use fewer or equal nodes than the axial (Manhattan) path.
    assert!(
        out_diag.routes[0].nodes.len() <= out_axial.routes[0].nodes.len(),
        "diagonal route should use fewer nodes: diag={} axial={}",
        out_diag.routes[0].nodes.len(),
        out_axial.routes[0].nodes.len()
    );
}
