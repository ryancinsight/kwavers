//! The routing resource model: per-node occupancy, congestion history, and adjacency.
//!
//! Each grid node is a routing resource with unit capacity. Negotiated-congestion routing lets
//! nets temporarily exceed capacity (sharing), then charges progressively for it via the present
//! and history terms until the routing is legal (every node at or under capacity).

use crate::geom::{GridSpec, Nm};

/// A flat grid-node index `(ix, iy, layer)` packed by [`GridSpec::node_index`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub usize);

/// Unit capacity of a routing node (one net per cell per layer).
pub(crate) const CAPACITY: u16 = 1;

/// A move from one node to an adjacent node.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    /// The neighbour reached.
    pub node: NodeId,
    /// Whether the move is a layer transition (a via).
    pub via: bool,
    /// Whether this move is a diagonal (NE/NW/SE/SW) in-plane move.
    /// Diagonal moves traverse √2 × the grid pitch; callers should scale the geometric
    /// base cost accordingly while keeping location-based physics penalties unscaled.
    pub diagonal: bool,
}

/// The routing grid and its mutable congestion state.
#[derive(Debug, Clone)]
pub struct Grid {
    spec: GridSpec,
    occ: Vec<u16>,
    hist: Vec<f32>,
    blocked: Vec<bool>,
    /// Per-node net id of the (last) net occupying it, or `NO_OWNER`. Drives **width-aware
    /// clearance**: foreign copper within [`clearance_radius`] cells of a routed cell is a clearance
    /// violation, so the same cell can carry a net's own copper (radius is intra-layer, same-net
    /// exempt) while repelling other nets' copper that the bare grid pitch is too fine to separate.
    owner: Vec<i32>,
    /// Clearance halo radius in cells: a routed cell needs every other net's copper ≥ `radius + 1`
    /// cells away (in-plane, same layer). `0` ⇒ the grid pitch already *is* the clearance quantum
    /// (track + clearance ≤ pitch), so one-net-per-cell suffices and the halo is a no-op — the
    /// coarse-grid behaviour, bit-for-bit unchanged. `> 0` on a sub-clearance fine grid.
    clearance_radius: usize,
    /// Per in-plane column `(ix, iy)`: the net id owning a via there, or `-1` if none. A via's
    /// hole + annular ring occupies the whole layer column, so a column with a foreign via is a
    /// keepout for every other net on every layer.
    via_owner: Vec<i32>,
    /// Mathematically precise Euclidean via-to-via clearance limit.
    via_clearance_limit: Option<Nm>,
    /// Whether diagonal (45°) in-plane moves are enabled. Default `false`.
    /// Enabled at the routing-grid level when the design rules allow 45° routing.
    diagonal_routing: bool,
}

/// Sentinel for "no via in this column".
pub(crate) const NO_VIA: i32 = -1;
/// Sentinel for "no net owns this node".
pub(crate) const NO_OWNER: i32 = -1;

impl Grid {
    /// Build an empty grid (no occupancy, no history, nothing blocked) over a spec.
    #[must_use]
    pub fn new(spec: GridSpec) -> Self {
        let n = spec.len();
        Grid {
            spec,
            occ: vec![0; n],
            hist: vec![0.0; n],
            blocked: vec![false; n],
            owner: vec![NO_OWNER; n],
            clearance_radius: 0,
            via_owner: vec![NO_VIA; spec.nx * spec.ny],
            via_clearance_limit: None,
            diagonal_routing: false,
        }
    }

    /// Set the width-aware-clearance halo radius (cells a foreign net's copper must stay away from
    /// each routed cell, in-plane on the same layer). `0` keeps the one-net-per-cell coarse-grid
    /// behaviour. The caller derives it from `ceil((track + clearance) / pitch) - 1`.
    pub fn set_clearance_radius(&mut self, radius: usize) {
        self.clearance_radius = radius;
    }

    /// The clearance halo radius in cells (`0` ⇒ pitch is the clearance quantum).
    #[must_use]
    pub fn clearance_radius(&self) -> usize {
        self.clearance_radius
    }

    /// Set the Euclidean via-to-via clearance limit.
    pub fn set_via_clearance_limit(&mut self, limit: Nm) {
        self.via_clearance_limit = Some(limit);
    }

    /// Enable or disable diagonal (45°) in-plane routing moves.
    pub fn set_diagonal_routing(&mut self, enabled: bool) {
        self.diagonal_routing = enabled;
    }

    /// Whether diagonal (45°) in-plane moves are enabled.
    #[must_use]
    pub fn diagonal_routing(&self) -> bool {
        self.diagonal_routing
    }

    /// Net id owning a routed node, or `NO_OWNER` when the node is empty.
    #[inline]
    #[must_use]
    pub fn owner(&self, node: NodeId) -> i32 {
        self.owner[node.0]
    }

    /// How much **foreign** copper sits inside this node's clearance halo: the count of cells within
    /// `clearance_radius` (Chebyshev, in-plane on the node's layer, excluding the node itself) that
    /// are occupied by a net other than `net`. `0` when the halo is disabled (`clearance_radius == 0`)
    /// or the node is clear — the negotiated-clearance load that a fine grid adds to a cell's cost.
    #[inline]
    #[must_use]
    pub fn foreign_halo_load(&self, node: NodeId, net: i32) -> u32 {
        let r = self.clearance_radius;
        if r == 0 {
            return 0;
        }
        let (ix, iy, layer) = self.spec.node_coords(node.0);
        let (nx, ny) = (self.spec.nx as i64, self.spec.ny as i64);
        let (r, ix, iy) = (r as i64, ix as i64, iy as i64);
        let mut load = 0;
        for dy in -r..=r {
            let cy = iy + dy;
            if cy < 0 || cy >= ny {
                continue;
            }
            for dx in -r..=r {
                if dx == 0 && dy == 0 {
                    continue; // the node itself is handled by occupancy, not the halo
                }
                let cx = ix + dx;
                if cx < 0 || cx >= nx {
                    continue;
                }
                let idx = self.spec.node_index(cx as usize, cy as usize, layer);
                if self.occ[idx] > 0 && self.owner[idx] != net && self.owner[idx] != NO_OWNER {
                    load += 1;
                }
            }
        }
        load
    }

    /// In-plane column index `iy*nx + ix` of a node.
    #[inline]
    #[must_use]
    pub fn column_of(&self, node: NodeId) -> usize {
        let (ix, iy, _) = self.spec.node_coords(node.0);
        iy * self.spec.nx + ix
    }

    /// The net owning a via in a node's column, or `NO_VIA`.
    #[inline]
    #[must_use]
    pub fn via_owner(&self, node: NodeId) -> i32 {
        self.via_owner[self.column_of(node)]
    }

    /// Whether placing a via at `node`'s column would sit in, or orthogonally adjacent to, a column
    /// already holding a *foreign* net's via — i.e. the two annular rings would clash. Used to keep
    /// different-net vias a clear cell apart (the via-spacing rule).
    #[inline]
    #[must_use]
    pub fn near_foreign_via(&self, node: NodeId, net: i32) -> bool {
        let (ix, iy, _) = self.spec.node_coords(node.0);
        let (nx, ny) = (self.spec.nx as i64, self.spec.ny as i64);
        if let Some(limit) = self.via_clearance_limit {
            let pitch = self.spec.pitch;
            let limit_nm = limit.0 as i128;
            let pitch_nm = pitch.0 as i128;
            let limit_sq = limit_nm * limit_nm;
            let r_cells = ((limit_nm as f64) / (pitch_nm as f64)).ceil() as i64;
            for dy in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let (cx, cy) = (ix as i64 + dx, iy as i64 + dy);
                    if cx >= 0 && cx < nx && cy >= 0 && cy < ny {
                        let owner = self.via_owner[cy as usize * self.spec.nx + cx as usize];
                        if owner != NO_VIA && owner != net {
                            let dist_sq = (dx as i128 * pitch_nm) * (dx as i128 * pitch_nm)
                                + (dy as i128 * pitch_nm) * (dy as i128 * pitch_nm);
                            if dist_sq < limit_sq {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        } else {
            for (dx, dy) in [(0i64, 0i64), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (cx, cy) = (ix as i64 + dx, iy as i64 + dy);
                if cx >= 0 && cx < nx && cy >= 0 && cy < ny {
                    let owner = self.via_owner[cy as usize * self.spec.nx + cx as usize];
                    if owner != NO_VIA && owner != net {
                        return true;
                    }
                }
            }
            false
        }
    }

    /// Record that `net` placed a via in this node's column.
    #[inline]
    pub fn set_via(&mut self, node: NodeId, net: i32) {
        let c = self.column_of(node);
        self.via_owner[c] = net;
    }

    /// Clear a via record from a node's column.
    #[inline]
    pub fn clear_via(&mut self, node: NodeId) {
        let c = self.column_of(node);
        self.via_owner[c] = NO_VIA;
    }

    /// The grid specification.
    #[must_use]
    pub fn spec(&self) -> &GridSpec {
        &self.spec
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.occ.len()
    }

    /// Whether the grid is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.occ.is_empty()
    }

    /// Mark a node as a hard keepout (e.g. a foreign-net pad or a board cut-out).
    pub fn block(&mut self, node: NodeId) {
        self.blocked[node.0] = true;
    }

    /// Reserve a copper-free margin around the board outline: block every cell whose centre lies
    /// within `margin` of any board edge, on every layer. Best-practice edge clearance so no
    /// routed copper reaches the routed/V-scored outline.
    pub fn reserve_edge(&mut self, margin: crate::geom::Nm) {
        let pitch = self.spec.pitch.0;
        if pitch <= 0 || margin.0 <= 0 {
            return;
        }
        let mc = ((margin.0 + pitch - 1) / pitch) as usize; // ceil(margin / pitch)
        if mc == 0 {
            return;
        }
        for layer in 0..self.spec.nlayers {
            for iy in 0..self.spec.ny {
                let edge_y = iy < mc || iy + mc >= self.spec.ny;
                for ix in 0..self.spec.nx {
                    if edge_y || ix < mc || ix + mc >= self.spec.nx {
                        let n = self.spec.node_index(ix, iy, layer);
                        self.blocked[n] = true;
                    }
                }
            }
        }
    }

    /// Whether a node is a hard keepout.
    #[must_use]
    pub fn is_blocked(&self, node: NodeId) -> bool {
        self.blocked[node.0]
    }

    /// Current occupancy of a node.
    #[inline]
    #[must_use]
    pub fn occupancy(&self, node: NodeId) -> u16 {
        self.occ[node.0]
    }

    /// Over-capacity amount of a node (`0` when legal).
    #[inline]
    #[must_use]
    pub fn overuse(&self, node: NodeId) -> u16 {
        self.occ[node.0].saturating_sub(CAPACITY)
    }

    /// History cost accumulated on a node.
    #[inline]
    #[must_use]
    pub fn history(&self, node: NodeId) -> f32 {
        self.hist[node.0]
    }

    /// Claim one unit of a node for `net`, recording it as the node's owner (for width-aware
    /// clearance — a foreign net's copper within the halo of this node then registers as a violation).
    #[inline]
    pub fn claim(&mut self, node: NodeId, net: i32) {
        self.occ[node.0] += 1;
        self.owner[node.0] = net;
    }

    /// Release one unit of a node previously claimed; clears ownership once the node is empty so a
    /// stale owner cannot keep repelling foreign copper from a freed cell.
    #[inline]
    pub fn release(&mut self, node: NodeId) {
        self.occ[node.0] = self.occ[node.0].saturating_sub(1);
        if self.occ[node.0] == 0 {
            self.owner[node.0] = NO_OWNER;
        }
    }

    /// Add `delta` to a node's history cost (called once per iteration on over-used nodes).
    #[inline]
    pub fn bump_history(&mut self, node: NodeId, delta: f32) {
        self.hist[node.0] += delta;
    }

    /// Total number of distinct over-capacity nodes — the legality metric (0 ⇒ legal).
    #[must_use]
    pub fn overused_nodes(&self) -> usize {
        self.occ.iter().filter(|&&o| o > CAPACITY).count()
    }

    /// Compute a per-node over-capacity flag vector. `result[i]` is `true` iff `NodeId(i)`
    /// exceeds `CAPACITY` (occupancy > 1). Used by the targeted rip-up heuristic in the
    /// PathFinder loop to identify which nets must be re-routed each iteration.
    #[must_use]
    pub fn overuse_bitset(&self) -> Vec<bool> {
        self.occ.iter().map(|&o| o > CAPACITY).collect()
    }

    /// The per-iteration PathFinder history update, extended for width-aware clearance. Charges
    /// history on two kinds of illegal node and returns the count of distinct illegal nodes (`0` ⇒
    /// the routing is legal):
    /// * **over-capacity** — more than one net on the cell (the classic congestion term);
    /// * **clearance-violating** — a routed cell with foreign copper inside its clearance halo
    ///   (only possible when `clearance_radius > 0`, i.e. a sub-clearance fine grid).
    ///
    /// `decay` removes a fraction of the history from nodes that are **no longer** illegal each
    /// iteration, preventing stale congestion penalties from permanently biasing the router away
    /// from channels that have since cleared. Pass `0.0` for classic PathFinder (no decay).
    pub fn accumulate_history(&mut self, gain: f32, decay: f32) -> usize {
        let mut count = 0;
        for i in 0..self.occ.len() {
            let overuse = self.occ[i].saturating_sub(CAPACITY);
            let clearance_bad = self.clearance_radius > 0
                && self.occ[i] > 0
                && self.foreign_halo_load(NodeId(i), self.owner[i]) > 0;
            if overuse > 0 || clearance_bad {
                // Charge at least one unit of clearance pressure even when occupancy is at capacity,
                // so a clearance-only violation still escalates toward separation.
                let pressure = overuse.max(clearance_bad as u16);
                self.hist[i] += gain * pressure as f32;
                count += 1;
            } else if decay > 0.0 {
                // Decay history on nodes that are no longer congested, keeping penalties fresh
                // and preventing stale gradients from blocking cleared channels in later iterations.
                self.hist[i] *= 1.0 - decay;
            }
        }
        count
    }

    /// Per in-plane column `(ix, iy)`, the total negotiated-congestion history summed over layers.
    /// High where the router fought congestion — the signal fed back to congestion-driven
    /// placement so the next placement pulls components out of those regions.
    #[must_use]
    pub fn congestion_field(&self) -> Vec<f32> {
        let mut f = vec![0.0f32; self.spec.nx * self.spec.ny];
        for node in 0..self.hist.len() {
            let (ix, iy, _) = self.spec.node_coords(node);
            f[iy * self.spec.nx + ix] += self.hist[node];
        }
        f
    }

    /// Adjacency: the in-plane 4-neighbourhood plus the two inter-layer via moves. Blocked nodes
    /// are omitted, so a hard keepout is never even enumerated.
    /// When `diagonal_routing` is enabled, the four 45° diagonal in-plane moves are also included.
    pub fn neighbors(&self, node: NodeId, out: &mut Vec<Step>) {
        out.clear();
        let (ix, iy, layer) = self.spec.node_coords(node.0);
        // In-plane axial moves.
        if ix + 1 < self.spec.nx {
            self.push(self.spec.node_index(ix + 1, iy, layer), false, false, out);
        }
        if ix > 0 {
            self.push(self.spec.node_index(ix - 1, iy, layer), false, false, out);
        }
        if iy + 1 < self.spec.ny {
            self.push(self.spec.node_index(ix, iy + 1, layer), false, false, out);
        }
        if iy > 0 {
            self.push(self.spec.node_index(ix, iy - 1, layer), false, false, out);
        }
        // Via moves.
        if layer + 1 < self.spec.nlayers {
            self.push(self.spec.node_index(ix, iy, layer + 1), true, false, out);
        }
        if layer > 0 {
            self.push(self.spec.node_index(ix, iy, layer - 1), true, false, out);
        }
        // Diagonal moves (NE, NW, SE, SW).
        if self.diagonal_routing {
            if ix + 1 < self.spec.nx && iy + 1 < self.spec.ny {
                self.push(
                    self.spec.node_index(ix + 1, iy + 1, layer),
                    false,
                    true,
                    out,
                );
            }
            if ix + 1 < self.spec.nx && iy > 0 {
                self.push(
                    self.spec.node_index(ix + 1, iy - 1, layer),
                    false,
                    true,
                    out,
                );
            }
            if ix > 0 && iy + 1 < self.spec.ny {
                self.push(
                    self.spec.node_index(ix - 1, iy + 1, layer),
                    false,
                    true,
                    out,
                );
            }
            if ix > 0 && iy > 0 {
                self.push(
                    self.spec.node_index(ix - 1, iy - 1, layer),
                    false,
                    true,
                    out,
                );
            }
        }
    }

    fn push(&self, idx: usize, via: bool, diagonal: bool, out: &mut Vec<Step>) {
        if !self.blocked[idx] {
            out.push(Step {
                node: NodeId(idx),
                via,
                diagonal,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Nm;

    #[test]
    fn interior_node_has_four_planar_and_two_via_neighbours() {
        let spec =
            GridSpec::cover(Nm::from_mm(4.0), Nm::from_mm(4.0), Nm::from_mm(1.0), 4).unwrap();
        let g = Grid::new(spec);
        let mid = NodeId(spec.node_index(2, 2, 1));
        let mut nb = Vec::new();
        g.neighbors(mid, &mut nb);
        assert_eq!(nb.iter().filter(|s| !s.via && !s.diagonal).count(), 4);
        assert_eq!(nb.iter().filter(|s| s.via).count(), 2);
    }

    #[test]
    fn blocked_neighbour_is_not_enumerated() {
        let spec =
            GridSpec::cover(Nm::from_mm(4.0), Nm::from_mm(4.0), Nm::from_mm(1.0), 1).unwrap();
        let mut g = Grid::new(spec);
        let center = NodeId(spec.node_index(2, 2, 0));
        let right = NodeId(spec.node_index(3, 2, 0));
        g.block(right);
        let mut nb = Vec::new();
        g.neighbors(center, &mut nb);
        assert!(!nb.iter().any(|s| s.node == right));
    }

    #[test]
    fn reserve_edge_blocks_the_perimeter_ring() {
        let spec =
            GridSpec::cover(Nm::from_mm(5.0), Nm::from_mm(5.0), Nm::from_mm(0.5), 2).unwrap();
        let mut g = Grid::new(spec);
        g.reserve_edge(Nm::from_mm(1.0)); // 1.0 mm / 0.5 mm pitch => 2-cell ring
                                          // Corner and near-edge cells are blocked on every layer.
        for layer in 0..spec.nlayers {
            assert!(g.is_blocked(NodeId(spec.node_index(0, 0, layer))));
            assert!(g.is_blocked(NodeId(spec.node_index(1, 3, layer))));
            assert!(g.is_blocked(NodeId(spec.node_index(spec.nx - 1, 2, layer))));
        }
        // An interior cell well inside the margin is free.
        let cx = spec.nx / 2;
        let cy = spec.ny / 2;
        assert!(!g.is_blocked(NodeId(spec.node_index(cx, cy, 0))));
        // A neighbour enumeration from just inside the ring never yields a blocked edge cell.
        let mut nb = Vec::new();
        g.neighbors(NodeId(spec.node_index(2, 2, 0)), &mut nb);
        assert!(nb.iter().all(|s| !g.is_blocked(s.node)));
    }

    #[test]
    fn diagonal_neighbour_is_not_enumerated_by_default() {
        let spec =
            GridSpec::cover(Nm::from_mm(4.0), Nm::from_mm(4.0), Nm::from_mm(1.0), 1).unwrap();
        let g = Grid::new(spec);
        let mid = NodeId(spec.node_index(2, 2, 0));
        let mut nb = Vec::new();
        g.neighbors(mid, &mut nb);
        // Default: 4 axial only (no diagonals).
        assert_eq!(nb.iter().filter(|s| !s.via && !s.diagonal).count(), 4);
        assert_eq!(nb.iter().filter(|s| s.diagonal).count(), 0);
    }

    #[test]
    fn diagonal_routing_adds_four_diagonal_neighbours() {
        let spec =
            GridSpec::cover(Nm::from_mm(4.0), Nm::from_mm(4.0), Nm::from_mm(1.0), 1).unwrap();
        let mut g = Grid::new(spec);
        g.set_diagonal_routing(true);
        let mid = NodeId(spec.node_index(2, 2, 0));
        let mut nb = Vec::new();
        g.neighbors(mid, &mut nb);
        assert_eq!(
            nb.iter().filter(|s| !s.via && !s.diagonal).count(),
            4,
            "still 4 axial"
        );
        assert_eq!(
            nb.iter().filter(|s| s.diagonal).count(),
            4,
            "4 diagonal added"
        );
    }

    #[test]
    fn overuse_tracks_capacity() {
        let spec =
            GridSpec::cover(Nm::from_mm(2.0), Nm::from_mm(2.0), Nm::from_mm(1.0), 1).unwrap();
        let mut g = Grid::new(spec);
        let n = NodeId(0);
        assert_eq!(g.overuse(n), 0);
        g.claim(n, 0);
        assert_eq!(g.overuse(n), 0);
        g.claim(n, 1);
        assert_eq!(g.overuse(n), 1);
        assert_eq!(g.overused_nodes(), 1);
        g.release(n);
        assert_eq!(g.overuse(n), 0);
    }

    #[test]
    fn overuse_bitset_flags_only_overused_nodes() {
        let spec =
            GridSpec::cover(Nm::from_mm(2.0), Nm::from_mm(2.0), Nm::from_mm(1.0), 1).unwrap();
        let mut g = Grid::new(spec);
        let n0 = NodeId(0);
        let n1 = NodeId(1);
        g.claim(n0, 0);
        g.claim(n0, 1); // n0 overused: 2 claims > CAPACITY=1
        g.claim(n1, 2); // n1 legal: 1 claim ≤ CAPACITY
        let bits = g.overuse_bitset();
        assert!(bits[n0.0], "node 0 must be flagged overused");
        assert!(!bits[n1.0], "node 1 must not be flagged overused");
        // Releasing one claim on n0 brings it back to legal.
        g.release(n0);
        let bits2 = g.overuse_bitset();
        assert!(!bits2[n0.0], "node 0 must be legal after release");
    }

    #[test]
    fn accumulate_history_decays_cleared_nodes() {
        let spec =
            GridSpec::cover(Nm::from_mm(2.0), Nm::from_mm(2.0), Nm::from_mm(1.0), 1).unwrap();
        let mut g = Grid::new(spec);
        let n = NodeId(0);
        // Drive the node overused: two different nets on the same cell.
        g.claim(n, 0);
        g.claim(n, 1);
        assert_eq!(g.overuse(n), 1, "node must be overused");
        // Accumulate: history rises on the overused node.
        g.accumulate_history(1.0, 0.1);
        let h_after_overuse = g.history(n);
        assert!(
            h_after_overuse > 0.0,
            "history must increase on an overused node"
        );
        // Clear the overuse by releasing one claim.
        g.release(n);
        assert_eq!(g.overuse(n), 0, "node must be legal after release");
        // Accumulate with decay: history must shrink on the now-legal node.
        g.accumulate_history(1.0, 0.1);
        let h_after_decay = g.history(n);
        assert!(
            h_after_decay < h_after_overuse,
            "history must decay on a legal node: {h_after_decay:.4} < {h_after_overuse:.4}"
        );
        // With zero decay, history on a legal node is unchanged.
        let baseline = h_after_decay;
        g.accumulate_history(1.0, 0.0);
        assert_eq!(
            g.history(n),
            baseline,
            "zero decay must not change history on a legal node"
        );
    }
}
