//! Prim-style (power / ground) + chain-tip (signal / HV) tree-growth helpers for the
//! PathFinder routing loop.
//!
//! Phase 2b round-2 carved the single-net grow-loop body out of
//! [`crate::route::pathfinder::Router::route_with_obstacles`] into this module's
//! `impl Router { fn route_one }` block. The tree-growth dispatcher picks Prim-style vs
//! chain-tip by net class:
//!
//! * **Power / ground** use **Prim-style tree growth**: start from any terminal, expand
//!   the frontier outward via [`crate::route::search::grow_tree`], accept any reached
//!   terminal until all groups are covered. Cheap + permissive.
//! * **Signal / HV** use **chain-tip growth**: start from terminal 0, expand via
//!   `grow_tree`, and **replace the search source with each reached terminal's group** so
//!   multi-terminal high-speed nets form a *daisy chain* rather than branching. Chain-tip
//!   is the key invariant that prevents the test
//!   `multi_terminal_signal_routes_as_daisy_chain` from regressing to a stub / branch
//!   topology (`max degree ≤ 2` enforced by the test).
//!
//! (Plain backticks throughout for crate-internal references — the convention adopted
//! at Phase 2a to avoid rustdoc's `private_intra_doc_links` / `redundant_explicit_links`
//! lints. The router struct itself lives at
//! [`crate::route::pathfinder::Router`], declared in `pathfinder.rs`.)

use std::collections::HashSet;

use crate::board::NetClassKind;
use crate::cost::RoutingCost;
use crate::route::grid::NodeId;
use crate::route::pathfinder::{NetRoute, NetTerminals, Router};
use crate::route::search::{grow_tree, Scratch};

impl<C: RoutingCost> Router<C> {
    /// Route a single net. Power / ground nets use Prim-style tree growth. Signal / HV nets use
    /// chain-tip growth so multi-terminal high-speed nets prefer daisy-chain topology over stubs.
    ///
    /// `pub(super)` so [`Router::route_with_obstacles`](crate::route::pathfinder::Router::route_with_obstacles)
    /// (in `pathfinder.rs`) can call it through the cross-file `impl Router` block split.
    pub(super) fn route_one(
        &self,
        net: &NetTerminals,
        forbidden: &HashSet<NodeId>,
        via_forbidden: &HashSet<NodeId>,
        present: f64,
        scratch: &mut Scratch,
    ) -> NetRoute {
        let mut route = NetRoute::default();
        if net.terminal_groups.is_empty() {
            route.connected = true;
            return route;
        }
        let mut tree: HashSet<NodeId> = HashSet::new();
        for &node in &net.terminal_groups[0] {
            if tree.insert(node) {
                route.nodes.push(node);
            }
        }

        let mut remaining_groups: Vec<usize> = (1..net.terminal_groups.len()).collect();
        let mut chain_sources: HashSet<NodeId> = tree.clone();
        let high_speed_chain = matches!(net.class, NetClassKind::Hv | NetClassKind::Signal);
        route.connected = true;

        while !remaining_groups.is_empty() {
            let remaining: HashSet<NodeId> = remaining_groups
                .iter()
                .flat_map(|&group| net.terminal_groups[group].iter().copied())
                .filter(|node| !tree.contains(node))
                .collect();
            if remaining.is_empty() {
                break;
            }
            let sources = if high_speed_chain {
                &chain_sources
            } else {
                &tree
            };
            let chain_forbidden;
            let search_forbidden = if high_speed_chain {
                chain_forbidden = {
                    let mut f = forbidden.clone();
                    for node in tree.difference(&chain_sources) {
                        f.insert(*node);
                    }
                    f
                };
                &chain_forbidden
            } else {
                forbidden
            };
            let path = grow_tree(
                &self.grid,
                &self.cost,
                net.class,
                net.net.0 as i32,
                sources,
                &remaining,
                search_forbidden,
                via_forbidden,
                present,
                scratch,
            );
            let Some(path) = path else {
                // No remaining terminal reachable: net stays partially connected.
                route.connected = false;
                break;
            };
            // Graft the path onto the tree, recording new nodes and edges.
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                route.edges.push((a, b));
                if tree.insert(b) {
                    route.nodes.push(b);
                }
            }
            // Any reached logical terminal contributes its whole access group: a drilled pad is a
            // physical barrel, so the router need not emit an artificial via inside the pad.
            let mut reached_groups = Vec::new();
            remaining_groups.retain(|&group| {
                let reached = net.terminal_groups[group]
                    .iter()
                    .any(|node| tree.contains(node));
                if reached {
                    reached_groups.push(group);
                    for &node in &net.terminal_groups[group] {
                        if tree.insert(node) {
                            route.nodes.push(node);
                        }
                    }
                }
                !reached
            });
            if high_speed_chain {
                if let Some(&group) = reached_groups.first() {
                    chain_sources = net.terminal_groups[group].iter().copied().collect();
                }
            }
        }
        route
    }
}
