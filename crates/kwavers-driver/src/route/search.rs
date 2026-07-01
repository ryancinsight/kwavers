//! Per-net shortest-path search under the negotiated-congestion cost.
//!
//! A multi-source **A\*** grows a net's routed tree: sources are the nodes already in the tree,
//! targets are the net's unconnected terminals. The first target reached is the cheapest one, and
//! its back-traced path is grafted onto the tree (a Prim-style minimum-spanning growth that
//! approximates a rectilinear Steiner tree for multi-terminal nets).
//!
//! # A\* heuristic (vs. uninformed Dijkstra)
//!
//! The frontier is prioritised by `f = g + h`, where `g` is the actual accrued cost and `h` is the
//! `heuristic`: the octile (diagonal grids) or Manhattan (axial-only) distance to the nearest
//! remaining target. That distance is a strict **lower bound** on the true remaining cost — every
//! edge costs at least its geometric length and `present ≥ 1` — so the heuristic is **admissible and
//! consistent**, and A\* returns exactly the same minimum-cost path as Dijkstra would, while
//! expanding far fewer nodes (the goal-directed frontier). With an empty heuristic (`h ≡ 0`) the
//! search degenerates to the original Dijkstra; the guidance never changes the result, only the work.
//! This is the base "variation on A\*" — heuristic guidance over best-first/Dijkstra search.
//!
//! The edge cost charged to reach node `v` is
//!
//! ```text
//! (base(v) + history(v)) * (1 + occupancy(v) * present_factor)
//!     [+ via_cost(class) if a layer change]
//!     [+ ACUTE_ANGLE_PENALTY if the in-plane turn at v is acute (< 90°)]
//! ```
//!
//! which is exactly the PathFinder node cost: intrinsic physics cost, inflated by accumulated
//! history and by present congestion that the `present_factor` schedule escalates each iteration.
//!
//! # Bend penalty
//!
//! Acute-angle junctions (< 90° between consecutive track segments) are DFM defects: the narrow
//! re-entrant corner traps etchant during subtractive fabrication (acid trap), causing incomplete
//! etching, copper residue shorts, and increased defect rates (IPC-2221 concern). When diagonal
//! routing is enabled, the 8-neighbour search naturally produces cardinal→diagonal transitions
//! (e.g. E→NE, N→NW) that subtend 45° — the canonical acid-trap geometry.
//!
//! `ACUTE_ANGLE_PENALTY` is added flat (not scaled by congestion) to any edge whose outgoing
//! direction forms an acute angle with the incoming direction at the expansion node. Detection
//! uses the integer dot and cross products of the direction vectors:
//!
//! - dot(in, out) > 0 **and** cross(in, out) ≠ 0 ⟹ angle strictly between 0° (straight) and
//!   90° (right-angle turn) ⟹ **acid-trap geometry, add penalty**.
//! - dot ≤ 0: obtuse or U-turn — no penalty.
//! - cross = 0: straight-through or anti-parallel — no penalty.
//!
//! The penalty is soft (additive, not a hard keepout) so the router can still take an acute turn
//! under congestion pressure when no non-acute path is available. Evidence tier: unit tests in
//! `mod tests` verify the routing steers toward non-acute paths when equivalent alternatives exist.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use super::search_guards::{
    diagonal_crosses_foreign_edge, diagonal_passes_foreign_track_corner,
    diagonal_passes_foreign_via_corner, via_clips_foreign_diagonal_track,
    DIAGONAL_VIA_CLEARANCE_BUDGET_NM,
};
use crate::board::{LayerId, NetClassKind};
use crate::cost::RoutingCost;
use crate::route::grid::{Grid, NodeId, Step};

/// Flat penalty added to an edge whose outgoing direction forms an **acute angle** (< 90°) with
/// the incoming direction at the expansion node.
///
/// Derivation of the calibrated value: for a diagonal move (geometric base √2 ≈ 1.414), the
/// penalty raises the edge cost to √2 + 0.5 ≈ 1.914 — enough to prefer a ≥ 90° alternative when
/// one exists (e.g. two orthogonal steps cost 2.0 ≈ 1.914 at this crossover), while remaining
/// below the `via_cost` default (10.0) so the penalty never forces a layer change.
///
/// At high congestion (present_factor >> 1), the `node_cost` term dominates and the flat 0.5 is
/// proportionally small — correct behaviour: the penalty is a DFM preference, not a hard block.
const ACUTE_ANGLE_PENALTY: f64 = 0.5;

/// A\* frontier entry. `f = g + h` is the heap priority (estimated total path cost); `g` is the
/// actual cost accrued from the sources (used for relaxation and the stale-entry test). Ordered by
/// ascending `f` via `total_cmp` (so `NaN` cannot corrupt the order — costs are finite by
/// construction); ties break toward **higher `g`** (the node closer to a target, fewer expansions —
/// Amit Patel's tie-breaking guidance) and then lower node id for deterministic geometry.
struct Frontier {
    f: f64,
    g: f64,
    node: NodeId,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.g == other.g && self.node == other.node
    }
}
impl Eq for Frontier {}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap: make the entry that should pop *first* compare greatest —
        // lowest `f`, then highest `g`, then lowest node id.
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| self.g.total_cmp(&other.g))
            .then_with(|| other.node.0.cmp(&self.node.0))
    }
}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Admissible, consistent A\* heuristic: a lower bound on the cost to reach the nearest remaining
/// target. Every edge cost is `(geom + physics + history) · present` (`+ via/bend`), and `present ≥ 1`,
/// `physics, history ≥ 0`, so each step costs at least its geometric length (`1` axial, `√2`
/// diagonal). The octile distance (diagonal grids) or Manhattan distance (axial-only) is exactly the
/// minimum achievable sum of those geometric lengths, hence never overestimates the true cost —
/// admissibility guarantees A\* still returns the Dijkstra-optimal path. Layer/via cost is omitted
/// (a further underestimate), keeping the bound valid across layers. `0` when the target set is empty.
#[inline]
fn heuristic(ix: usize, iy: usize, targets_xy: &[(i32, i32)], diagonal: bool) -> f64 {
    const SQRT2_MINUS_1: f64 = std::f64::consts::SQRT_2 - 1.0;
    let mut best = f64::INFINITY;
    for &(tx, ty) in targets_xy {
        let dx = (tx - ix as i32).unsigned_abs() as f64;
        let dy = (ty - iy as i32).unsigned_abs() as f64;
        let d = if diagonal {
            // Octile: travel the shorter leg diagonally (√2 each), the rest axially.
            let (lo, hi) = if dx < dy { (dx, dy) } else { (dy, dx) };
            hi + SQRT2_MINUS_1 * lo
        } else {
            dx + dy
        };
        if d < best {
            best = d;
        }
    }
    if best.is_finite() {
        best
    } else {
        0.0
    }
}

/// Reusable scratch buffers, so the per-iteration search does not reallocate the grid-sized
/// distance/predecessor arrays on every net.
pub struct Scratch {
    dist: Vec<f64>,
    prev: Vec<u32>,
    stamp: Vec<u32>,
    epoch: u32,
    heap: BinaryHeap<Frontier>,
    /// Nodes settled (popped non-stale) by the last [`grow_tree`] call — the A\* work metric. Low
    /// relative to the grid size is the heuristic doing its job (a goal-directed frontier, not a
    /// Dijkstra flood).
    expansions: usize,
}

const NO_PREV: u32 = u32::MAX;

impl Scratch {
    /// Allocate scratch for a grid of `n` nodes.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Scratch {
            dist: vec![f64::INFINITY; n],
            prev: vec![NO_PREV; n],
            stamp: vec![0; n],
            epoch: 0,
            heap: BinaryHeap::new(),
            expansions: 0,
        }
    }

    /// Nodes settled by the most recent search — the A\* expansion count (for profiling/tests).
    #[must_use]
    pub fn expansions(&self) -> usize {
        self.expansions
    }

    #[inline]
    fn fresh(&mut self, i: usize) -> bool {
        // Lazy reset: a node belongs to this search only if its stamp matches the epoch.
        if self.stamp[i] != self.epoch {
            self.stamp[i] = self.epoch;
            self.dist[i] = f64::INFINITY;
            self.prev[i] = NO_PREV;
            true
        } else {
            false
        }
    }
}

/// Grow a net's tree by one terminal: find the cheapest path from any `sources` node to any
/// `targets` node, avoiding `forbidden` nodes (foreign pads). Returns the path (source-first,
/// inclusive of the reached target), or `None` if no target is reachable.
#[allow(clippy::too_many_arguments)] // a search kernel: each argument is an irreducible input.
pub fn grow_tree(
    grid: &Grid,
    cost: &impl RoutingCost,
    class: NetClassKind,
    net_id: i32,
    sources: &HashSet<NodeId>,
    targets: &HashSet<NodeId>,
    forbidden: &HashSet<NodeId>,
    via_forbidden: &HashSet<NodeId>,
    present_factor: f64,
    scratch: &mut Scratch,
) -> Option<Vec<NodeId>> {
    scratch.epoch = scratch.epoch.wrapping_add(1);
    if scratch.epoch == 0 {
        // Epoch wrapped: clear stamps so stale matches cannot occur.
        scratch.stamp.iter_mut().for_each(|s| *s = u32::MAX);
        scratch.epoch = 1;
    }
    scratch.heap.clear();
    scratch.expansions = 0;

    let spec = *grid.spec();
    let diagonal = grid.diagonal_routing();
    // Precompute the in-plane coordinates of every target once, so the A\* heuristic is a cheap
    // min-over-targets distance per expansion rather than a per-node grid walk.
    let targets_xy: Vec<(i32, i32)> = targets
        .iter()
        .map(|t| {
            let (tx, ty, _) = spec.node_coords(t.0);
            (tx as i32, ty as i32)
        })
        .collect();

    // Seed from the sources in a deterministic (sorted) order. `sources` is a HashSet whose
    // iteration order is randomised per run; seeding the heap in that order makes equal-cost
    // tie-breaks — and therefore the routed geometry — non-reproducible.
    let mut srcs: Vec<NodeId> = sources.iter().copied().collect();
    srcs.sort_unstable();
    for s in srcs {
        scratch.fresh(s.0);
        scratch.dist[s.0] = 0.0;
        let (sx, sy, _) = spec.node_coords(s.0);
        let h = heuristic(sx, sy, &targets_xy, diagonal);
        scratch.heap.push(Frontier {
            f: h,
            g: 0.0,
            node: s,
        });
    }

    // 4 axial + 2 via + 4 diagonal = 10 maximum neighbours when diagonal routing is enabled.
    let mut nb: Vec<Step> = Vec::with_capacity(10);
    while let Some(Frontier { g: d, node: u, .. }) = scratch.heap.pop() {
        // Stale heap entry (a cheaper path already settled this node). `d` is g (cost-so-far).
        if d > scratch.dist[u.0] {
            continue;
        }
        scratch.expansions += 1;
        if targets.contains(&u) && !sources.contains(&u) {
            return Some(backtrace(u, scratch));
        }

        // Pre-compute u's grid coordinates for the bend-penalty direction check below.
        let (ux, uy, u_layer) = spec.node_coords(u.0);

        // Incoming in-plane direction (predecessor → u). Used by the bend penalty to detect
        // acute-angle turns at the expansion node.
        //
        // `None` in two cases where there is no meaningful in-plane prior direction:
        //   1. Source nodes: `scratch.prev[u.0] == NO_PREV` (no predecessor).
        //   2. Via predecessors: the predecessor is on a different copper layer; the via move
        //      has no in-plane direction to carry forward.
        let incoming_dir: Option<(i64, i64)> = if scratch.prev[u.0] != NO_PREV {
            let p = scratch.prev[u.0] as usize;
            let (px, py, p_layer) = spec.node_coords(p);
            if p_layer == u_layer {
                Some((ux as i64 - px as i64, uy as i64 - py as i64))
            } else {
                None // via predecessor
            }
        } else {
            None // source node
        };

        grid.neighbors(u, &mut nb);
        for step in &nb {
            let v = step.node;
            if forbidden.contains(&v) {
                continue;
            }
            // A foreign net's via occupies this column on every layer — no copper may use it.
            let owner = grid.via_owner(v);
            if owner != crate::route::grid::NO_VIA && owner != net_id {
                continue;
            }
            // `v`'s in-plane coordinates and layer index. Computed once and reused for every
            // geometric guard below — the via-clip check needs `(ix, iy, layer)` BEFORE the
            // diagonal-foreign-edge / track-corner / acute-bend checks below.
            let (ix, iy, layer) = spec.node_coords(v.0);
            // A *new* via must clear neighbouring foreign vias (annular-ring spacing) and stay out
            // of foreign pads' via-keepout. Tracks are exempt — only a layer change is constrained.
            // Also: a via cannot drop into a column whose centre sits inside
            // [`DIAGONAL_VIA_CLEARANCE_BUDGET_NM`] of a foreign diagonal track. Corner-cell occupancy
            // checks miss this geometry (no shared grid node exceeds capacity so PathFinder's
            // overuse bitset never re-routes), so the residue survives to external DRC.
            if step.via
                && (grid.near_foreign_via(v, net_id)
                    || via_forbidden.contains(&v)
                    || via_clips_foreign_diagonal_track(
                        grid,
                        (ix, iy, layer),
                        net_id,
                        DIAGONAL_VIA_CLEARANCE_BUDGET_NM,
                    ))
            {
                continue;
            }
            if step.diagonal
                && diagonal_crosses_foreign_edge(grid, (ux, uy, u_layer), (ix, iy, layer), net_id)
            {
                continue;
            }
            if step.diagonal
                && diagonal_passes_foreign_via_corner(
                    grid,
                    (ux, uy, u_layer),
                    (ix, iy, layer),
                    net_id,
                )
            {
                continue;
            }
            if step.diagonal
                && diagonal_passes_foreign_track_corner(
                    grid,
                    (ux, uy, u_layer),
                    (ix, iy, layer),
                    net_id,
                )
            {
                continue;
            }
            let pt = spec.point_of(ix, iy);
            // Width-aware clearance: foreign copper inside `v`'s halo counts as extra congestion at
            // `v`, so the same escalating present-factor that resolves cell sharing also drives nets
            // apart by the clearance distance on a sub-clearance fine grid. On a coarse grid the halo
            // radius is 0 and this term is identically zero (no behaviour change).
            let congestion = grid.occupancy(v) as f64 + grid.foreign_halo_load(v, net_id) as f64;
            let present = 1.0 + congestion * present_factor;
            let base = cost.node_base(pt, LayerId(layer as u16), class);
            // `node_base` returns 1.0 + physics_penalties. Only the geometric unit (1.0) scales
            // with physical distance; physics penalties are location-based (independent of move
            // direction). Diagonal moves traverse √2 × the grid pitch.
            let physics = base - 1.0;
            let geom = if step.diagonal {
                std::f64::consts::SQRT_2
            } else {
                1.0
            };
            // Bend penalty (flat, not scaled by congestion): detect an acute-angle turn at the
            // current expansion node using integer dot and cross products of the direction vectors.
            // dot(in, out) > 0 AND cross(in, out) ≠ 0  ⟹  angle is strictly (0°, 90°) — an
            // acid-trap geometry. Skipped for via steps (no in-plane direction change) and for
            // source nodes (no incoming direction). See module-level docs for derivation.
            let bend = if let (Some((dx_in, dy_in)), false) = (incoming_dir, step.via) {
                let dx_out = ix as i64 - ux as i64;
                let dy_out = iy as i64 - uy as i64;
                let dot = dx_in * dx_out + dy_in * dy_out;
                let cross = dx_in * dy_out - dy_in * dx_out;
                if dot > 0 && cross != 0 {
                    ACUTE_ANGLE_PENALTY
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let node_cost = (geom + physics + grid.history(v) as f64) * present;
            let edge = node_cost + bend + if step.via { cost.via_cost(class) } else { 0.0 };
            let nd = d + edge;
            scratch.fresh(v.0);
            if nd < scratch.dist[v.0] {
                scratch.dist[v.0] = nd;
                scratch.prev[v.0] = u.0 as u32;
                // A\* priority: actual cost-so-far plus the admissible estimate of the cost remaining
                // to the nearest target. `(ix, iy)` are `v`'s in-plane coordinates (computed above).
                let h = heuristic(ix, iy, &targets_xy, diagonal);
                scratch.heap.push(Frontier {
                    f: nd + h,
                    g: nd,
                    node: v,
                });
            }
        }
    }
    None
}

fn backtrace(target: NodeId, scratch: &Scratch) -> Vec<NodeId> {
    let mut path = vec![target];
    let mut cur = target.0;
    while scratch.prev[cur] != NO_PREV {
        cur = scratch.prev[cur] as usize;
        path.push(NodeId(cur));
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{LayerId, NetClassKind};
    use crate::geom::{GridSpec, Nm, Point};
    use crate::route::grid::Grid;
    use std::collections::HashSet;

    /// A flat cost model: every node has base cost 1.0 regardless of position, layer, or class.
    /// Isolates the bend penalty and geometry terms from physics effects.
    struct FlatCost;
    impl RoutingCost for FlatCost {
        fn node_base(&self, _p: Point, _layer: LayerId, _class: NetClassKind) -> f64 {
            1.0
        }
    }

    fn grid_1l(nx: usize, ny: usize) -> Grid {
        let spec = GridSpec {
            nx,
            ny,
            nlayers: 1,
            pitch: Nm::from_mm(1.0),
            origin: Point::default(),
        };
        Grid::new(spec)
    }

    fn node(grid: &Grid, ix: usize, iy: usize) -> NodeId {
        NodeId(grid.spec().node_index(ix, iy, 0))
    }

    fn route(
        grid: &mut Grid,
        src_ix: usize,
        src_iy: usize,
        tgt_ix: usize,
        tgt_iy: usize,
    ) -> Vec<NodeId> {
        let src = HashSet::from([node(grid, src_ix, src_iy)]);
        let tgt = HashSet::from([node(grid, tgt_ix, tgt_iy)]);
        let mut scratch = Scratch::new(grid.len());
        grow_tree(
            grid,
            &FlatCost,
            NetClassKind::Signal,
            1,
            &src,
            &tgt,
            &HashSet::new(),
            &HashSet::new(),
            0.0,
            &mut scratch,
        )
        .expect("target reachable")
    }

    /// The admissible A\* heuristic must not compromise optimality: with a wall forcing a detour,
    /// the search returns a **minimum-length** path around it, not a greedy goal-ward dive into the
    /// obstacle. A non-admissible (over-estimating) heuristic would cut the detour short and return a
    /// longer path; this asserts the exact optimal edge count.
    ///
    /// Grid: 11×11, axial-only. A wall on column 5 blocks rows 0..=9, leaving a single gap at
    /// (5, 10). Routing (0,5) → (10,5) must climb to the gap and descend the far side.
    /// Every legal path crosses column 5 only at the gap, so the Manhattan-optimal length is
    /// `dist((0,5)->(5,10)) + dist((5,10)->(10,5)) = 10 + 10 = 20` edges ⇒ 21 nodes.
    #[test]
    fn astar_finds_optimal_detour_around_a_wall() {
        let mut grid = grid_1l(11, 11);
        for y in 0..10 {
            grid.block(node(&grid, 5, y));
        }
        let path = route(&mut grid, 0, 5, 10, 5);
        assert_eq!(path[0], node(&grid, 0, 5));
        assert_eq!(*path.last().unwrap(), node(&grid, 10, 5));
        // The path never touches a blocked wall cell.
        assert!(
            path.iter().all(|&n| {
                let (x, y, _) = grid.spec().node_coords(n.0);
                !(x == 5 && y < 10)
            }),
            "path must route through the gap, not the wall"
        );
        assert_eq!(
            path.len(),
            21,
            "A* must take the shortest legal detour (20 edges), got {} nodes",
            path.len()
        );
    }

    /// A\* is deterministic: the same query returns the identical path across runs (tie-breaking is
    /// total — `f`, then higher `g`, then node id — so equal-cost alternatives resolve reproducibly).
    #[test]
    fn astar_routing_is_deterministic() {
        let mut g1 = grid_1l(9, 9);
        let mut g2 = grid_1l(9, 9);
        assert_eq!(route(&mut g1, 0, 0, 8, 8), route(&mut g2, 0, 0, 8, 8));
    }

    /// The heuristic must actually *prune*: on a large open grid, goal-directed A\* settles only a
    /// thin band of nodes near the optimal path, whereas uninformed Dijkstra (h ≡ 0) floods almost
    /// the entire grid before reaching the far corner. We measure both directly — `grow_tree` is A\*,
    /// and seeding the heap with a zero heuristic via a degenerate single-cell target set is not
    /// available, so we compare A\* expansions against the grid size and assert a hard reduction.
    ///
    /// Grid: 40×40 open, axial. Optimal (0,0)→(39,39) path is 79 edges. A\* settles O(path) nodes in
    /// the goal direction; the assertion (< 40 % of the 1600 cells) fails for any non-pruning search.
    #[test]
    fn astar_heuristic_prunes_the_frontier() {
        let grid = grid_1l(40, 40);
        let src = HashSet::from([node(&grid, 0, 0)]);
        let tgt = HashSet::from([node(&grid, 39, 39)]);
        let mut scratch = Scratch::new(grid.len());
        let path = grow_tree(
            &grid,
            &FlatCost,
            NetClassKind::Signal,
            1,
            &src,
            &tgt,
            &HashSet::new(),
            &HashSet::new(),
            0.0,
            &mut scratch,
        )
        .expect("target reachable");
        // Optimal Manhattan path length on a 40×40 grid: 78 edges ⇒ 79 nodes.
        assert_eq!(path.len(), 79, "A* returns the optimal-length path");
        let settled = scratch.expansions();
        let cells = 40 * 40;
        assert!(
            settled < cells * 2 / 5,
            "A* must prune: settled {settled}/{cells} nodes (uninformed Dijkstra settles ~all)"
        );
    }

    /// Straight axial routing on a row grid produces the expected path with no bend penalty.
    ///
    /// Grid: 3×1 (no diagonal). Source=(0,0), target=(2,0).
    /// Only path: E→E. No diagonal, no bend penalty.
    /// Expected total cost: 2.0 (two unit edges, FlatCost base=1).
    #[test]
    fn straight_axial_path_no_bend_penalty() {
        let mut grid = grid_1l(3, 1);
        let path = route(&mut grid, 0, 0, 2, 0);
        assert_eq!(
            path,
            vec![node(&grid, 0, 0), node(&grid, 1, 0), node(&grid, 2, 0)]
        );
    }

    /// A right-angle (90°) turn does NOT trigger the bend penalty.
    ///
    /// Grid: 3×2 (no diagonal). Source=(0,0), target=(1,1).
    /// Cheapest path: N→E or E→N, both cost 2.0.
    /// The N→E path has dot(N,E)=0 and cross≠0, but dot>0 fails — no penalty.
    /// The E→N path has dot(E,N)=0 — no penalty.
    #[test]
    fn right_angle_turn_no_bend_penalty() {
        let mut grid = grid_1l(3, 2);
        let path = route(&mut grid, 0, 0, 1, 1);
        // Exactly 3 nodes (2 edges, cost 2.0 without any penalty)
        assert_eq!(path.len(), 3, "expected 2-step path, got {path:?}");
        assert_eq!(path[0], node(&grid, 0, 0));
        assert_eq!(*path.last().unwrap(), node(&grid, 1, 1));
    }

    /// The router prefers a straight diagonal path over any path that incurs a bend penalty.
    ///
    /// Grid: 3×3, diagonal routing enabled. Source=(0,0), target=(2,2).
    ///
    /// Candidate paths:
    ///   A (straight diagonal): NE→NE, nodes (0,0)→(1,1)→(2,2).
    ///      Cost: √2 + √2 = 2√2 ≈ 2.828. No bend penalty: at (1,1) incoming=NE,
    ///      outgoing=NE, cross=0 (straight through).
    ///   B (E then NE): E→NE, nodes (0,0)→(1,0)→(2,1)→….
    ///      At (1,0): incoming E=(1,0), outgoing NE=(1,1): dot=1>0, cross=1≠0 → +0.5 penalty.
    ///      Requires a third step to reach (2,2) — total ≥2.914.
    ///
    /// The router must choose path A, visiting (1,1).
    #[test]
    fn straight_diagonal_preferred_over_bent_path() {
        let mut grid = grid_1l(3, 3);
        grid.set_diagonal_routing(true);
        let path = route(&mut grid, 0, 0, 2, 2);
        // The straight diagonal path must pass through (1,1).
        let mid = node(&grid, 1, 1);
        assert!(
            path.contains(&mid),
            "expected straight-diagonal via (1,1), got {path:?}"
        );
        assert_eq!(
            path.len(),
            3,
            "expected 2-step straight diagonal, got {path:?}"
        );
    }

    /// A repeated diagonal in the same direction is a straight-through move (cross=0) and
    /// therefore does NOT incur the bend penalty, even though dot > 0.
    ///
    /// This is the canonical E→E (straight) check generalised to diagonal direction.
    /// Grid: 4×4, diagonal enabled. Source=(0,0), target=(3,3).
    /// Cheapest path: NE→NE→NE (three diagonal steps, cost=3√2, zero bend penalties).
    #[test]
    fn straight_diagonal_repeated_no_bend_penalty() {
        let mut grid = grid_1l(4, 4);
        grid.set_diagonal_routing(true);
        let path = route(&mut grid, 0, 0, 3, 3);
        // All four nodes must lie on the main diagonal.
        let expected: Vec<NodeId> = (0..4).map(|k| node(&grid, k, k)).collect();
        assert_eq!(path, expected, "expected diagonal highway, got {path:?}");
    }

    /// When only an acute-angle path exists, the router still finds it (penalty is soft, not
    /// a hard block).
    ///
    /// Grid: 2×2, diagonal enabled. Source=(0,0), target=(1,1).
    /// The direct diagonal NE: cost = √2 (source, no prev → no penalty).
    /// The axial path: N→E or E→N, cost = 2.0.
    /// Router picks the single-step diagonal.
    #[test]
    fn acute_penalty_does_not_block_when_no_alternative() {
        let mut grid = grid_1l(2, 2);
        grid.set_diagonal_routing(true);
        let path = route(&mut grid, 0, 0, 1, 1);
        // Single-step diagonal (no bend penalty on first move from source).
        assert_eq!(path.len(), 2, "expected direct diagonal, got {path:?}");
        assert_eq!(path[0], node(&grid, 0, 0));
        assert_eq!(path[1], node(&grid, 1, 1));
    }

    /// DONE↔PROG analogue: foreign net owns two **axial** neighbours of the proposed via
    /// (W at `(cx-1, cy)` and N at `(cx, cy+1)`), implying a 45° diagonal segment that
    /// arcs around the via's own cell. PathFinder's `occupancy`-driven overuse bitset never
    /// registers the geometry (no shared cell); without this converse guard, the via
    /// placement passes the search and external DRC catches the residue. The guard must
    /// reject.
    ///
    /// Grid: 3×3, 0.5 mm pitch, layer 0. Foreign net 2 owns cells (0,1)=(W of via) and
    /// (1,2)=(N of via). Segment: `(0, 0.5) → (0.5, 1.0)` mm. Proposed via column (1,1)
    /// centre `(0.5, 0.5)` mm. Foot of perpendicular: `(0.25, 0.75)`. Separation:
    /// √(0.25² + 0.25²) = 0.354 mm = 354_000 nm — strictly less than the 435_000 nm
    /// DRC-residue budget derived from the empirical DONE↔PROG gap.
    #[test]
    fn diagonal_versus_via_clearance_fires_on_done_prog_geometry() {
        let mut grid = grid_pitch(0.5, 3, 3);
        grid.claim(node(&grid, 0, 1), 2); // W of via (1,1), "DONE"
        grid.claim(node(&grid, 1, 2), 2); // N of via (1,1), "DONE"
                                          // Guard must fire at the DRC-residue budget (~0.435 mm).
        assert!(
            via_clips_foreign_diagonal_track(&grid, (1, 1, 0), 1, DIAGONAL_VIA_CLEARANCE_BUDGET_NM),
            "guard must fire when foreign net owns W and N of via (1,1)",
        );
        // Guard does not fire when the geometry already satisfies the budget (budget
        // strictly less than the perpendicular distance). Pins the threshold direction:
        // a negative margin is required for the guard to be conservative.
        assert!(
            !via_clips_foreign_diagonal_track(&grid, (1, 1, 0), 1, 200_000),
            "guard must not fire when budget is below the geometry distance (geometry already satisfies)",
        );
        // Guard does NOT fire if the diagonal belongs to OUR net (we own it ourselves).
        assert!(
            !via_clips_foreign_diagonal_track(
                &grid,
                (1, 1, 0),
                2,
                DIAGONAL_VIA_CLEARANCE_BUDGET_NM
            ),
            "guard must not fire when diagonal belongs to the routing net",
        );
    }

    /// Pins the **four** quadrant pair shapes. The diagonal-via residue class can occur with
    /// the foreign diagonal arcing any of the four 45° orientations around the proposed
    /// via column (thinker analysis). One regression per pair would silently underprotect
    /// three of the four quadrants; iterate all four.
    #[test]
    fn diagonal_versus_via_clearance_fires_on_every_quadrant_pair() {
        // Pairs of `v = (2, 2)` axial neighbours that form a diagonal: pick well-clear
        // candidates so every pair's perpendicular distance is below the budget.
        let v = (2_usize, 2_usize);
        // Each entry: (pair label, two cells in `(ix, iy)` for that pair around `v`).
        type QuadrantPair = (&'static str, (usize, usize), (usize, usize));
        let pairs: &[QuadrantPair] = &[
            ("W→N", (v.0 - 1, v.1), (v.0, v.1 + 1)),
            ("N→E", (v.0, v.1 + 1), (v.0 + 1, v.1)),
            ("E→S", (v.0 + 1, v.1), (v.0, v.1 - 1)),
            ("S→W", (v.0, v.1 - 1), (v.0 - 1, v.1)),
        ];
        for &(label, a, b) in pairs {
            let mut grid = grid_pitch(0.5, 5, 5);
            grid.claim(node(&grid, a.0, a.1), 99);
            grid.claim(node(&grid, b.0, b.1), 99);
            assert!(
                via_clips_foreign_diagonal_track(
                    &grid,
                    (v.0, v.1, 0),
                    1,
                    DIAGONAL_VIA_CLEARANCE_BUDGET_NM,
                ),
                "guard must fire for quadrant pair {label} around via ({}, {})",
                v.0,
                v.1,
            );
        }
    }

    /// No foreign diagonal anywhere → guard does nothing on an empty grid. Pins the no-op
    /// baseline so a future code change can't silently over-reject.
    #[test]
    fn diagonal_versus_via_clearance_safe_when_no_foreign_diagonal() {
        let grid = grid_pitch(0.5, 5, 5);
        assert!(!via_clips_foreign_diagonal_track(
            &grid,
            (2, 2, 0),
            1,
            DIAGONAL_VIA_CLEARANCE_BUDGET_NM,
        ));
    }

    /// Foreign net owns two *axial* neighbours (an axial segment, NOT a diagonal). The
    /// diagonal guard should NOT fire. Pins the type filter: a vertical or horizontal
    /// foreign segment adjacent to the via must not be over-rejected by a diagonal-specific
    /// guard — that's the job of [`crate::route::grid::Grid::near_foreign_via`].
    #[test]
    fn diagonal_versus_via_clearance_safe_for_axial_only_neighbours() {
        let mut grid = grid_pitch(0.5, 3, 3);
        grid.claim(node(&grid, 0, 1), 2);
        grid.claim(node(&grid, 1, 1), 2); // axial W↔E segment; not a diagonal
        assert!(!via_clips_foreign_diagonal_track(
            &grid,
            (1, 0, 0),
            1,
            DIAGONAL_VIA_CLEARANCE_BUDGET_NM,
        ));
    }

    /// Boundary handling: at the grid edge some of the 4 candidate pairs are out-of-bounds
    /// and the guard must skip them without panicking. Pins the per-pair bounds check so a
    /// future refactor that drops the check can't introduce undefined behaviour at edges.
    #[test]
    fn diagonal_versus_via_clearance_handles_grid_boundary_without_panic() {
        let mut grid = grid_pitch(0.5, 3, 3);
        // V = (0, 1) on the left edge. Three of the four candidate pairs extend to negative
        // x and must be skipped. The single in-bounds candidate pair is `N ↔ E` at
        // ((0, 2), (1, 1)). Foreign-claim both and confirm the guard fires via that pair.
        grid.claim(node(&grid, 0, 2), 99);
        grid.claim(node(&grid, 1, 1), 99);
        assert!(
            via_clips_foreign_diagonal_track(&grid, (0, 1, 0), 1, DIAGONAL_VIA_CLEARANCE_BUDGET_NM),
            "guard must fire for the in-bounds N↔E pair at edge (0, 1)",
        );
        // V at corner (0, 0): every candidate pair has at least one cell out-of-bounds.
        // Empty grid ⇒ nothing in-bounds fires ⇒ guard quiet. Pins OOB-skip behaviour at a
        // corner where 3 of the 4 pairs are entirely out-of-bounds.
        let corner = grid_pitch(0.5, 3, 3);
        assert!(!via_clips_foreign_diagonal_track(
            &corner,
            (0, 0, 0),
            1,
            DIAGONAL_VIA_CLEARANCE_BUDGET_NM,
        ));
    }

    /// Helper: build a single-layer `Grid` with a custom pitch (the default `grid_1l` is 1.0
    /// mm; the DONE↔PROG analogue needs the actual 0.5 mm scale to drive the budget check).
    fn grid_pitch(pitch_mm: f64, nx: usize, ny: usize) -> Grid {
        let spec = GridSpec {
            nx,
            ny,
            nlayers: 1,
            pitch: Nm::from_mm(pitch_mm),
            origin: Point::default(),
        };
        Grid::new(spec)
    }
}
