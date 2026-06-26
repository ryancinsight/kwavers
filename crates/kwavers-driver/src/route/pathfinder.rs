//! The negotiated-congestion routing loop (PathFinder).
//!
//! Nets negotiate shared grid nodes through iterative rip-up and re-route: each iteration raises
//! the cost of over-capacity nodes via accumulated history and an escalating present factor, until
//! no node exceeds capacity (legal routing). The loop is order-independent — unlike sequential A\*
//! it cannot deadlock.
//!
//! # Convergence optimizations
//!
//! **Targeted rip-up**: only nets that occupy at least one over-capacity node are ripped up and
//! re-routed each iteration. Legal nets stay in place, preserving their routes and eliminating
//! redundant search work — especially valuable in later iterations when only a small fraction of
//! nets remain congested.
//!
//! **History decay** (`history_decay`): a small fraction of the history is removed from nodes
//! that are no longer over-capacity each iteration, preventing stale penalties from permanently
//! biasing the router away from channels that have since cleared.
//!
//! **Stall-break**: when the overused-node count fails to improve for three or more consecutive
//! iterations, the present factor receives a temporary multiplicative boost that breaks
//! congestion equilibria where two nets keep swapping the same overloaded resource.
//!
//! # Phase 2b round-2 carve-out
//!
//! Two concerns previously inlined here have been carved out into sibling modules (round-2):
//!
//! * `tree.rs` — the single-net `route_one` method (Prim-style for power / ground, chain-tip
//!   for signal / HV). The dispatcher logic (chain-sources tracking, terminal-group
//!   draining loop) lives in that file's `impl Router { fn route_one }` block.
//! * `emission.rs` — the `apply_to_board` track / via emit method on `Router`, plus the
//!   `via_nodes` / `via_shadow_nodes` free functions this file's `route_with_obstacles`
//!   uses for rip-up + claim accounting during the negotiation loop.
//!
//! Keeping the three impl-block members (Negotiation loop / Tree grow / Emission) in
//! separate files lets the negotiation loop's body here stay focused on scheduling + rip-up
//! accounting; the per-net tree-grow dispatcher and the per-net copper-emit policy each
//! live in their own file. Cross-file `impl Router` blocks are a standard Rust idiom:
//! methods resolve through the type regardless of which file the impl block was written
//! in. The Router struct's `grid` / `cost` / `params` fields are `pub(super)` so sibling
//! impl blocks in `tree.rs` / `emission.rs` can access `self.grid` etc.; external crate
//! users see the struct shape only (the public methods are the API surface).
//!
//! (Plain backticks throughout for crate-internal references — same convention as the
//! other route/ files and the cost slice.)

use std::collections::HashSet;

use crate::board::{NetClassKind, NetId};
use crate::cost::RoutingCost;
use crate::route::emission::{via_nodes, via_shadow_nodes};
use crate::route::grid::{Grid, NodeId};

/// A foreign-pad clearance halo: the grid nodes that nets *other than* `net` must keep out of so
/// their copper holds clearance from this pad's physical extent.
#[derive(Debug, Clone)]
pub struct PadObstacle {
    /// The net that owns the pad — exempt from this keepout (its own copper may land on it). `None`
    /// for an **unconnected** pad: real copper with no net, so it is a keepout for *every* net (a
    /// track or via that crossed it would short to dead copper).
    pub net: Option<NetId>,
    /// Nets belonging to the same component as this pad, which are exempt from creepage rules against this pad.
    pub same_component_nets: Vec<NetId>,
    /// Track-clearance halo for low-voltage signal nets (normal clearance).
    pub nodes_signal: Vec<NodeId>,
    /// Creepage track-clearance halo for low-voltage signal nets.
    pub nodes_signal_creepage: Vec<NodeId>,
    /// Track-clearance halo for high-voltage nets (normal clearance).
    pub nodes_hv: Vec<NodeId>,
    /// Creepage track-clearance halo for high-voltage nets.
    pub nodes_hv_creepage: Vec<NodeId>,
    /// Track-clearance halo for power rails (normal clearance).
    pub nodes_power: Vec<NodeId>,
    /// Creepage track-clearance halo for power rails.
    pub nodes_power_creepage: Vec<NodeId>,
    /// Wider via keepout: nodes a foreign *via* must avoid (annular ring > track width).
    pub via_keepout: Vec<NodeId>,
    /// Whether this pad is *drilled* (thru-hole). A drilled pad's via keepout applies to **every**
    /// net including its own, because hole-to-hole drill spacing is net-agnostic — a same-net via
    /// abutting a thru-hole pad still violates it.
    pub drilled: bool,
    /// Shrunken keepout halo for unconnected or large/thermal pads (geometry_guard only).
    pub nodes_shrunken: Vec<NodeId>,
    /// Whether this pad belongs to an HV (`NetClassKind::Hv`) net. Used to suppress the full
    /// `nodes_hv` keepout when an HV routing net passes a same-component HV pad: no creepage
    /// separation is needed between two HV signals on the same IC, so only the geometry guard
    /// (shrunken) halo is required, keeping inter-pad routing channels open.
    pub is_hv_pad: bool,
    /// Whether this pad has a large area (> 16 mm²).
    pub is_large_pad: bool,
}

/// A net's routing job: its electrical class and the grid nodes its pads map to.
#[derive(Debug, Clone)]
pub struct NetTerminals {
    /// The net being routed.
    pub net: NetId,
    /// Electrical class (drives cost and track width).
    pub class: NetClassKind,
    /// Logical terminals the net must connect. Each inner vector is one pad's equivalent access
    /// nodes: SMD pads have one node, while drilled pads expose the same barrel on every copper
    /// layer. The router may reach any node in the group and then treats the whole group as the
    /// pad's existing vertical copper.
    pub terminal_groups: Vec<Vec<NodeId>>,
}

/// A routed net: the set of grid nodes it occupies and the edges (consecutive node pairs) that
/// form its tree. Edges on one layer become tracks; edges across layers become vias.
#[derive(Debug, Clone, Default)]
pub struct NetRoute {
    /// Distinct nodes the net occupies (claimed for congestion accounting).
    pub nodes: Vec<NodeId>,
    /// Tree edges as `(from, to)` node pairs.
    pub edges: Vec<(NodeId, NodeId)>,
    /// Whether every terminal was connected.
    pub connected: bool,
}

/// Tunable schedule for the negotiation.
#[derive(Debug, Clone, Copy)]
pub struct PathFinderParams {
    /// Maximum negotiation iterations before giving up.
    pub max_iter: usize,
    /// Initial present-congestion factor.
    pub present0: f64,
    /// Per-iteration multiplier on the present-congestion factor (escalation).
    pub present_mul: f64,
    /// History accumulation gain per iteration.
    pub history_gain: f32,
    /// Per-iteration fractional decay applied to the history of nodes that are no longer
    /// over-capacity. Prevents stale congestion penalties from permanently biasing the router
    /// away from channels that have since cleared. `0.0` disables decay (classic PathFinder).
    /// Default: `0.05` (5 % per iteration — halves the penalty in ≈ 14 iterations of clearance).
    pub history_decay: f32,
}

impl Default for PathFinderParams {
    fn default() -> Self {
        PathFinderParams {
            max_iter: 40,
            present0: 0.5,
            present_mul: 1.6,
            history_gain: 0.5,
            history_decay: 0.05,
        }
    }
}

fn route_priority(class: NetClassKind) -> u8 {
    match class {
        NetClassKind::Hv => 0,
        NetClassKind::Signal => 1,
        NetClassKind::Power => 2,
        NetClassKind::Ground => 3,
    }
}

/// Outcome of a routing run.
#[derive(Debug, Clone)]
pub struct RouteOutcome {
    /// One route per input net (same order).
    pub routes: Vec<NetRoute>,
    /// Whether the routing is legal: no grid node exceeds capacity.
    pub legal: bool,
    /// Whether every net connected all its terminals.
    pub complete: bool,
    /// Number of over-capacity nodes remaining (0 when `legal`).
    pub overused_nodes: usize,
    /// Iterations actually run.
    pub iterations: usize,
}

/// The router: owns the grid resource model and the (physics-guided) cost.
///
/// Field visibility is `pub(super)` so the cross-file `impl Router` blocks in
/// `tree.rs` + `emission.rs` can access `self.grid` / `self.cost` / `self.params`
/// during their `route_one` + `apply_to_board` impls. External crate users only see
/// the public methods (the API surface) — the struct's three fields are invisible to
/// them, same as in the pre-split pathfinder.rs.
pub struct Router<C: RoutingCost> {
    pub(super) grid: Grid,
    pub(super) cost: C,
    pub(super) params: PathFinderParams,
}

impl<C: RoutingCost> Router<C> {
    /// Build a router over a grid with a cost model and the default schedule.
    pub fn new(grid: Grid, cost: C) -> Self {
        Router {
            grid,
            cost,
            params: PathFinderParams::default(),
        }
    }

    /// Override the negotiation schedule.
    #[must_use]
    pub fn with_params(mut self, params: PathFinderParams) -> Self {
        self.params = params;
        self
    }

    /// Borrow the underlying grid (for inspection / blocking foreign keepouts before routing).
    pub fn grid_mut(&mut self) -> &mut Grid {
        &mut self.grid
    }

    /// Borrow the underlying grid (e.g. to read the congestion field back for co-optimization).
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Route all nets to convergence (legal) or `max_iter`. Foreign pads are hard keepouts per net.
    pub fn route(&mut self, nets: &[NetTerminals]) -> RouteOutcome {
        self.route_with_obstacles(nets, &[])
    }

    /// Like [`Router::route`], but each net additionally avoids the clearance **halo** of every
    /// foreign pad. A pad is physical copper, not a point; routing one grid cell away can still
    /// violate clearance, so the caller inflates each pad to the cells it must keep other nets out
    /// of ([`PadObstacle`]). A net never avoids its *own* pads' halos (it must reach them).
    ///
    /// The single-net `route_one` body lives in `tree.rs` (Method resolution finds it through
    /// the cross-file `impl Router` block split regardless of which file the impl block was
    /// written in).
    pub fn route_with_obstacles(
        &mut self,
        nets: &[NetTerminals],
        obstacles: &[PadObstacle],
    ) -> RouteOutcome {
        // `Scratch` is allocated once here and reused across all iterations and all nets via the
        // epoch mechanism: each `grow_tree` call bumps the epoch, lazily invalidating per-node
        // state without clearing the full arrays. No grid-sized reallocation occurs inside the
        // routing loop.
        let mut scratch = crate::route::search::Scratch::new(self.grid.len());

        // Per-net forbidden sets: every terminal of every *other* net, plus foreign pad halos.
        let own: Vec<HashSet<NodeId>> = nets
            .iter()
            .map(|n| n.terminal_groups.iter().flatten().copied().collect())
            .collect();
        let all_terminals: HashSet<NodeId> = nets
            .iter()
            .flat_map(|n| n.terminal_groups.iter().flatten().copied())
            .collect();
        let forbidden: Vec<HashSet<NodeId>> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let mut f: HashSet<NodeId> = all_terminals.difference(&own[i]).copied().collect();
                for ob in obstacles {
                    // Foreign pad (different net) or unconnected pad (`net: None`) ⇒ keep clear.
                    if ob.net != Some(n.net) {
                        let is_same_component = ob.same_component_nets.contains(&n.net);
                        // Unconnected pads (`ob.net = None`) are dead copper: any track that
                        // physically overlaps them creates a short, so they always use the
                        // class-appropriate full halo — never the shrunken geometry-guard-only
                        // halo. Large same-component thermal pads (netted) may use the shrunken
                        // halo because same-net copper may legitimately land on them.
                        //
                        // Same-component HV pad vs. HV routing net: no creepage requirement
                        // exists between two HV signals on the same IC (creepage rules apply
                        // between HV and LV copper, not within the HV domain). Using the full
                        // `nodes_hv` halo would completely block the inter-pad routing channels
                        // between adjacent HV pads at 0.8 mm pitch (halo 0.325 mm each side
                        // > half-pitch 0.4 mm), making VPP/VNN unroutable. Use `nodes_shrunken`
                        // (geometry-guard only, 0.07 mm) so adjacent cells at 0.5 mm stay open;
                        // physical clearance = 0.5 - 0.175 (pad half-w) - 0.125 (track half-w)
                        // = 0.200 mm ≥ 0.13 mm rule. ✓
                        //
                        // Exception: drilled (PTH) pads — drill-hole-to-copper clearance is
                        // always required by IPC-2221 regardless of net class. A through-hole
                        // connector pad on the same component (e.g. J2 with TX_0..TX_15) must
                        // never use the shrunken halo; TX routes must respect the full HV
                        // clearance around every PTH barrel.
                        let is_same_component_hv = is_same_component
                            && n.class == NetClassKind::Hv
                            && ob.is_hv_pad
                            && !ob.drilled;
                        let nodes = if is_same_component && ob.is_large_pad && ob.net.is_some() {
                            &ob.nodes_shrunken
                        } else if is_same_component_hv {
                            // Same-component HV-to-HV: only the geometry guard keeps us off the
                            // pad copper itself; the inter-pad channels must remain open.
                            &ob.nodes_shrunken
                        } else {
                            match n.class {
                                NetClassKind::Signal => {
                                    if is_same_component {
                                        &ob.nodes_signal
                                    } else {
                                        &ob.nodes_signal_creepage
                                    }
                                }
                                // Ground tracks are power_track width (0.25 mm), not signal
                                // width (0.15 mm). Using nodes_signal would undersize the halo
                                // and allow GND tracks to come within 0.087 mm of adjacent HV
                                // pads on the same IC (VNN/VPP). Use nodes_power for the same
                                // clearance margin as other 0.25 mm-wide nets.
                                NetClassKind::Ground => {
                                    if is_same_component {
                                        &ob.nodes_power
                                    } else {
                                        &ob.nodes_power_creepage
                                    }
                                }
                                NetClassKind::Hv => {
                                    if is_same_component {
                                        &ob.nodes_hv
                                    } else {
                                        &ob.nodes_hv_creepage
                                    }
                                }
                                NetClassKind::Power => {
                                    if is_same_component {
                                        &ob.nodes_power
                                    } else {
                                        &ob.nodes_power_creepage
                                    }
                                }
                            }
                        };
                        f.extend(nodes.iter().copied());
                    }
                }
                // Ensure a net's own terminals are never forbidden
                for own_node in &own[i] {
                    f.remove(own_node);
                }
                f
            })
            .collect();
        // Wider keepout applied to *via placement only* (a via's ring exceeds a track), so tracks
        // can still use the channel between pads while vias stay clear of foreign pads.
        let via_forbidden: Vec<HashSet<NodeId>> = nets
            .iter()
            .map(|n| {
                let mut f = HashSet::new();
                for ob in obstacles {
                    // Foreign pads: keep this net's vias clear (annular-ring clearance). Drilled
                    // pads: keep *every* net's vias clear (net-agnostic hole-to-hole spacing).
                    if ob.net != Some(n.net) || ob.drilled {
                        f.extend(ob.via_keepout.iter().copied());
                    }
                }
                f
            })
            .collect();

        let mut routes: Vec<NetRoute> = vec![NetRoute::default(); nets.len()];
        let mut legal = false;
        let mut overused = 0;
        let mut iterations = 0;
        let spec = *self.grid.spec();
        // Sort by class priority first (HV > Signal > Power > Ground — more critical nets claim
        // preferred channels before simpler signals). Within the same class, sort descending by
        // terminal-group count: nets with more terminals are harder to route and benefit from
        // first access to uncongested resources. `idx` as final tiebreaker gives deterministic order.
        let mut route_order: Vec<usize> = (0..nets.len()).collect();
        route_order.sort_by_key(|&idx| {
            (
                route_priority(nets[idx].class),
                std::cmp::Reverse(nets[idx].terminal_groups.len()),
                idx,
            )
        });

        // Stall detection: count consecutive iterations where overused-node count does not
        // improve. Used to trigger a temporary schedule boost that breaks congestion equilibria
        // where two nets keep exchanging the same overloaded resource indefinitely.
        let mut prev_overused = usize::MAX;
        let mut stall_count = 0u32;

        for iter in 0..self.params.max_iter {
            // Base present factor following the exponential schedule.
            let base_present = self.params.present0 * self.params.present_mul.powi(iter as i32);
            // Stall-break multiplier: doubles every 3 consecutive non-improving iterations,
            // capped at 8× the base. Raising the present factor when stuck escalates the penalty
            // on persistently-overused nodes faster than the fixed schedule alone.
            let stall_boost = if stall_count >= 3 {
                2.0_f64.powi((stall_count / 3) as i32).min(8.0)
            } else {
                1.0
            };
            let present = base_present * stall_boost;

            // Targeted rip-up: on the first iteration every route is default/empty, so all nets
            // need initial routing. On subsequent iterations, compute the over-capacity bitset
            // once and only rip up nets that occupy at least one overused node — legal nets stay
            // in place, preserving their routes and eliminating redundant search work.
            let overuse_bits: Vec<bool> = if iter == 0 {
                vec![] // empty sentinel: first pass — route all nets unconditionally
            } else {
                self.grid.overuse_bitset()
            };

            for &i in &route_order {
                // A net needs re-routing when: it has no route yet (first pass or default),
                // or at least one of its grid nodes is over-capacity. Skipping legal nets is a
                // convergence optimisation — it never prevents legalisation because the history
                // mechanism ensures persistently-overused nodes are penalised until they clear.
                let needs_reroute = overuse_bits.is_empty()
                    || routes[i].nodes.is_empty()
                    || routes[i].nodes.iter().any(|n| overuse_bits[n.0]);
                if !needs_reroute {
                    continue; // Net is fully legal — leave its route in place.
                }

                let net = &nets[i];
                // Rip up this net's previous occupancy, via-column shadows, and via ownership.
                for &n in &routes[i].nodes {
                    self.grid.release(n);
                }
                for sn in via_shadow_nodes(&routes[i], &spec) {
                    self.grid.release(sn);
                }
                for vn in via_nodes(&routes[i], &spec) {
                    self.grid.clear_via(vn);
                }
                // `route_one` is `pub(super)` in `tree.rs`'s `impl Router` block; rust resolves it
                // through the cross-file method table.
                let route =
                    self.route_one(net, &forbidden[i], &via_forbidden[i], present, &mut scratch);
                let nid = net.net.0 as i32;
                for &n in &route.nodes {
                    self.grid.claim(n, nid);
                }
                for sn in via_shadow_nodes(&route, &spec) {
                    self.grid.claim(sn, nid);
                }
                for vn in via_nodes(&route, &spec) {
                    self.grid.set_via(vn, net.net.0 as i32);
                }
                routes[i] = route;
            }

            overused = self
                .grid
                .accumulate_history(self.params.history_gain, self.params.history_decay);
            iterations = iter + 1;

            // Update stall tracking for the next iteration's schedule boost decision.
            if overused < prev_overused {
                stall_count = 0;
            } else {
                stall_count = stall_count.saturating_add(1);
            }
            prev_overused = overused;

            if overused == 0 {
                legal = true;
                break;
            }
        }

        let complete = routes.iter().all(|r| r.connected);
        RouteOutcome {
            routes,
            legal,
            complete,
            overused_nodes: overused,
            iterations,
        }
    }
}
