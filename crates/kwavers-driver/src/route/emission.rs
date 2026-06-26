//! Track + via emission from a `RouteOutcome` onto a `Board`, plus the via-column helpers.
//!
//! Phase 2b round-2 carved three concerns out of `src/route/pathfinder.rs` into this
//! sibling module:
//!
//! * `via_nodes` — endpoints of via edges (layer-changing edges) on the lower-indexed
//!   node. Both endpoints of a via edge share the same `(ix, iy)`; the helper reports the
//!   `a` (lower-indexed) endpoint so the via's column can be claimed / released.
//! * `via_shadow_nodes` — every layer-node in a via's column that the route does NOT
//!   already occupy. Claiming these in the [`crate::route::grid::Grid`] makes a foreign
//!   track in the through-hole column register as an occupancy conflict, forcing the
//!   negotiation to push it out (without the shadow, a foreign track in a through-hole's
//!   column would still be physically blocked by the barrel but survive the search
//!   legality check). Excludes the net's own nodes to avoid self-overuse.
//! * `impl Router { fn apply_to_board }` — emits routed tracks + vias onto a
//!   [`crate::board::Board`] using the per-class widths from the
//!   [`crate::rules::DesignRules::track_for`] helper and the per-design via policy from
//!   [`crate::rules::DesignRules::resolve_via`]. A via spans exactly the layer range the
//!   net occupies at that column (never an artificial through-hole that would clip
//!   foreign tracks on free layers).
//!
//! (Plain backticks throughout for crate-internal references — same convention as
//! `tree.rs` and the cost slice. The router struct itself lives at
//! [`crate::route::pathfinder::Router`], declared in `pathfinder.rs`.)

use std::collections::HashSet;

use crate::board::{Board, LayerId, Track, Via};
use crate::cost::RoutingCost;
use crate::geom::GridSpec;
use crate::route::grid::NodeId;
use crate::route::pathfinder::{NetRoute, NetTerminals, RouteOutcome, Router};
use crate::rules::DesignRules;

/// The endpoints of a route's via edges (layer-changing edges), reported on the
/// lower-indexed node so the via's column can be claimed / released. Both edge
/// endpoints share the same `(ix, iy)`.
pub(super) fn via_nodes(route: &NetRoute, spec: &GridSpec) -> Vec<NodeId> {
    route
        .edges
        .iter()
        .filter_map(|&(a, b)| {
            let (_, _, la) = spec.node_coords(a.0);
            let (_, _, lb) = spec.node_coords(b.0);
            (la != lb).then_some(a)
        })
        .collect()
}

/// The "shadow" nodes of a route's vias: every layer-node in a via's column that the
/// route does not already occupy. A via is a through-hole — it physically blocks its
/// whole column — so claiming these makes a foreign track in the column register as an
/// occupancy conflict (counted in legality), forcing the negotiation to push it out.
/// Excludes the net's own nodes to avoid self-overuse.
pub(super) fn via_shadow_nodes(route: &NetRoute, spec: &GridSpec) -> Vec<NodeId> {
    let own: HashSet<NodeId> = route.nodes.iter().copied().collect();
    let mut cols: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for vn in via_nodes(route, spec) {
        let (ix, iy, _) = spec.node_coords(vn.0);
        cols.insert(iy * spec.nx + ix);
    }
    let mut out = Vec::new();
    for col in cols {
        let (ix, iy) = (col % spec.nx, col / spec.nx);
        for layer in 0..spec.nlayers {
            let n = NodeId(spec.node_index(ix, iy, layer));
            if !own.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

impl<C: RoutingCost> Router<C> {
    /// Emit the routed tracks and vias of an outcome onto a board, using the design-rule
    /// widths. Test surface in `src/route/tests.rs` calls this directly to verify the
    /// end-to-end contract (HV-tile routing dumps real copper tracks + vias).
    pub fn apply_to_board(
        &self,
        board: &mut Board,
        nets: &[NetTerminals],
        outcome: &RouteOutcome,
        rules: &DesignRules,
    ) {
        let spec = *self.grid.spec();
        for (net, route) in nets.iter().zip(&outcome.routes) {
            let width = rules.track_for(net.class);
            // Per column, the min/max layer this net occupies — a via spans exactly
            // that range (never artificially through-hole, which would clip foreign
            // tracks on free layers).
            let mut col_span: std::collections::BTreeMap<usize, (usize, usize)> =
                std::collections::BTreeMap::new();
            let mut via_cols: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            for &n in &route.nodes {
                let (ix, iy, l) = spec.node_coords(n.0);
                let e = col_span.entry(iy * spec.nx + ix).or_insert((l, l));
                e.0 = e.0.min(l);
                e.1 = e.1.max(l);
            }
            for &(a, b) in &route.edges {
                let (ax, ay, al) = spec.node_coords(a.0);
                let (bx, by, bl) = spec.node_coords(b.0);
                if al == bl {
                    board.tracks.push(Track {
                        start: spec.point_of(ax, ay),
                        end: spec.point_of(bx, by),
                        width,
                        layer: LayerId(al as u16),
                        net: net.net,
                    });
                } else {
                    via_cols.insert(ay * spec.nx + ax);
                }
            }
            for col in via_cols {
                let (ix, iy) = (col % spec.nx, col / spec.nx);
                let (lo, hi) = col_span.get(&col).copied().unwrap_or((0, spec.nlayers - 1));
                // Build the via per the board's via policy: a standard through-hole on
                // a standard stackup, or the actual span classified into HDI
                // micro / blind / buried under an HDI stackup. The router only emits a
                // via where the net changes layer; the *construction* class is the
                // design's choice, not implied by the span alone.
                let (from, to, kind, drill, diameter) =
                    rules.resolve_via(lo as u16, hi as u16, spec.nlayers);
                board.vias.push(Via {
                    pos: spec.point_of(ix, iy),
                    drill,
                    diameter,
                    net: net.net,
                    from,
                    to,
                    kind,
                    filled: false,
                });
            }
        }
    }
}
