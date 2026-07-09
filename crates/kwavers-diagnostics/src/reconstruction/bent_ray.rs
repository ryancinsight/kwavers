//! Bent-ray (Fermat) traveltime tomography via shortest-path ray tracing.
//!
//! In heterogeneous media a transmission ray **refracts**: its path is the one of
//! least traveltime (Fermat's principle), bending toward fast (low-slowness)
//! regions. Straight-ray Radon/FBP (ADR 013) and the straight-line acoustic
//! projection mismodel this geometry; the bias grows with `|∇c|/c`.
//!
//! This module traces the Fermat ray as the **shortest path** through an
//! 8-connected grid graph over a slowness field `s(r) = 1/c(r)`, with edge cost
//! `½(s_u + s_v)·L_{uv}` (trapezoidal slowness × Euclidean edge length). The
//! minimum-cost path is the bent ray; its per-voxel accumulated path length is
//! exactly the system-matrix row of the discretized line integral
//! `t = ∫ s dℓ = Σ_v s_v · row_v`, so it plugs straight into the existing
//! SIRT/ART/OSEM reconstructor as the bent-ray forward operator.
//!
//! See ADR 020. Straight-ray remains the fast default (`radon`).
//!
//! # References
//! - Moser, T. J. (1991). "Shortest path calculation of seismic rays."
//!   *Geophysics*, 56(1), 59–67.
//! - Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs."
//!   *Numer. Math.*, 1, 269–271.

use leto::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A bent (refracted) ray traced through a slowness field by Fermat shortest path.
#[derive(Debug, Clone)]
pub struct BentRay {
    /// Total traveltime along the Fermat path [s].
    pub traveltime: f64,
    /// Voxel indices `(i, j)` visited along the path, source → receiver.
    pub path: Vec<(usize, usize)>,
    /// Sparse system-matrix row `(flattened_voxel_index, accumulated_length_m)`
    /// with `flattened = i*ny + j`. Satisfies `traveltime == Σ s_v · length_v`
    /// exactly for the slowness field that produced the path.
    pub row: Vec<(usize, f64)>,
}

/// Min-heap entry ordered by ascending tentative cost (finite, non-negative).
struct HeapNode {
    cost: f64,
    node: usize,
}
impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for HeapNode {}
impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) pops the *smallest* cost. Costs are
        // finite and non-negative, so partial_cmp never returns None here.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The eight grid-neighbour offsets and their Euclidean edge lengths (× `dx`).
const NEIGHBORS: [(isize, isize, f64); 8] = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, std::f64::consts::SQRT_2),
    (-1, 1, std::f64::consts::SQRT_2),
    (1, -1, std::f64::consts::SQRT_2),
    (1, 1, std::f64::consts::SQRT_2),
];

/// Trace the Fermat (least-traveltime) ray from `source` to `receiver` through a
/// 2-D `slowness` field `[s/m]` on an isotropic grid of spacing `dx` `[m]`.
///
/// Returns the bent ray (traveltime, voxel path, and the per-voxel path-length
/// system-matrix row). Returns `None` if either endpoint is out of bounds or the
/// receiver is unreachable (it always is on a connected grid with finite slowness).
///
/// # Panics
/// Does not panic; all slowness values are read as-is (callers should pass
/// finite, positive slowness).
#[must_use]
pub fn bent_ray_path(
    slowness: &Array2<f64>,
    dx: f64,
    source: (usize, usize),
    receiver: (usize, usize),
) -> Option<BentRay> {
    let (nx, ny) = slowness.dim();
    if source.0 >= nx || source.1 >= ny || receiver.0 >= nx || receiver.1 >= ny {
        return None;
    }
    let n = nx * ny;
    let flat = |i: usize, j: usize| i * ny + j;
    let src = flat(source.0, source.1);
    let dst = flat(receiver.0, receiver.1);

    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    dist[src] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapNode {
        cost: 0.0,
        node: src,
    });

    while let Some(HeapNode { cost, node }) = heap.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        if node == dst {
            break;
        }
        let i = node / ny;
        let j = node % ny;
        let s_u = slowness[[i, j]];
        for &(di, dj, len) in &NEIGHBORS {
            let ni = i as isize + di;
            let nj = j as isize + dj;
            if ni < 0 || nj < 0 || ni >= nx as isize || nj >= ny as isize {
                continue;
            }
            let (ni, nj) = (ni as usize, nj as usize);
            let nidx = flat(ni, nj);
            if visited[nidx] {
                continue;
            }
            // Trapezoidal slowness integral over the edge.
            let edge_cost = 0.5 * (s_u + slowness[[ni, nj]]) * len * dx;
            let nd = cost + edge_cost;
            if nd < dist[nidx] {
                dist[nidx] = nd;
                prev[nidx] = node;
                heap.push(HeapNode {
                    cost: nd,
                    node: nidx,
                });
            }
        }
    }

    if !dist[dst].is_finite() {
        return None;
    }

    // Back-trace source → receiver.
    let mut path_nodes = Vec::new();
    let mut cur = dst;
    while cur != usize::MAX {
        path_nodes.push(cur);
        if cur == src {
            break;
        }
        cur = prev[cur];
    }
    path_nodes.reverse();
    let path: Vec<(usize, usize)> = path_nodes.iter().map(|&p| (p / ny, p % ny)).collect();

    // Build the per-voxel path-length row: each traversed edge of length L·dx
    // contributes L·dx/2 to each of its two endpoint voxels, so
    // Σ_v s_v · row_v = Σ_edges ½(s_u+s_v)·L·dx == traveltime.
    let mut row_map: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for w in path_nodes.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (ai, aj) = (a / ny, a % ny);
        let (bi, bj) = (b / ny, b % ny);
        let di = (ai as isize - bi as isize).abs();
        let dj = (aj as isize - bj as isize).abs();
        let len = if di + dj == 2 {
            std::f64::consts::SQRT_2
        } else {
            1.0
        } * dx;
        *row_map.entry(a).or_insert(0.0) += 0.5 * len;
        *row_map.entry(b).or_insert(0.0) += 0.5 * len;
    }
    let mut row: Vec<(usize, f64)> = row_map.into_iter().collect();
    row.sort_unstable_by_key(|&(idx, _)| idx);

    Some(BentRay {
        traveltime: dist[dst],
        path,
        row,
    })
}

/// Convenience: the Fermat traveltime from `source` to `receiver` (see
/// [`bent_ray_path`]). Returns `None` on out-of-bounds endpoints.
#[must_use]
pub fn bent_ray_traveltime(
    slowness: &Array2<f64>,
    dx: f64,
    source: (usize, usize),
    receiver: (usize, usize),
) -> Option<f64> {
    bent_ray_path(slowness, dx, source, receiver).map(|r| r.traveltime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    /// Homogeneous medium, axis-aligned ray: the bent path is the straight grid
    /// row and the traveltime equals `s · distance` exactly.
    #[test]
    fn homogeneous_axis_aligned_is_exact() {
        let s0 = 1.0 / 1500.0; // s/m
        let dx = 1e-3;
        let slow = Array2::from_elem((16, 16), s0);
        let ray = bent_ray_path(&slow, dx, (8, 2), (8, 13)).expect("ray");
        let expected = s0 * (13 - 2) as f64 * dx;
        assert!(
            (ray.traveltime - expected).abs() <= 1e-15 + 1e-12 * expected,
            "axis traveltime {:.6e} != {:.6e}",
            ray.traveltime,
            expected
        );
        // Path stays on the source row.
        assert!(ray.path.iter().all(|&(i, _)| i == 8), "path left the row");
    }

    /// Homogeneous medium, 45° diagonal ray: also exactly representable on the
    /// 8-connected grid → `traveltime = s · √2 · n · dx`.
    #[test]
    fn homogeneous_diagonal_is_exact() {
        let s0 = 1.0 / 1500.0;
        let dx = 1e-3;
        let slow = Array2::from_elem((16, 16), s0);
        let ray = bent_ray_path(&slow, dx, (2, 2), (12, 12)).expect("ray");
        let expected = s0 * 10.0 * std::f64::consts::SQRT_2 * dx;
        assert!(
            (ray.traveltime - expected).abs() <= 1e-15 + 1e-12 * expected,
            "diagonal traveltime {:.6e} != {:.6e}",
            ray.traveltime,
            expected
        );
    }

    /// The system-matrix row reproduces the traveltime exactly:
    /// `Σ_v slowness_v · row_v == traveltime`.
    #[test]
    fn row_reproduces_traveltime() {
        let dx = 1e-3;
        let mut slow = Array2::from_elem((20, 20), 1.0 / 1500.0);
        // A heterogeneous patch so the path is non-trivial.
        for i in 6..14 {
            for j in 8..12 {
                slow[[i, j]] = 1.0 / 1800.0;
            }
        }
        let ray = bent_ray_path(&slow, dx, (3, 3), (16, 17)).expect("ray");
        let recomputed: f64 = ray
            .row
            .iter()
            .map(|&(idx, len)| slow[[idx / 20, idx % 20]] * len)
            .sum();
        assert!(
            (recomputed - ray.traveltime).abs() <= 1e-15 + 1e-12 * ray.traveltime,
            "row Σ s·ℓ {:.9e} != traveltime {:.9e}",
            recomputed,
            ray.traveltime
        );
    }

    /// Graph-metric bound: the bent-ray traveltime is never shorter than the
    /// straight Euclidean line in a homogeneous medium, and is within the
    /// 8-connectivity overestimate bound.
    #[test]
    fn graph_metric_bounds_straight_line() {
        let s0 = 1.0 / 1500.0;
        let dx = 1e-3;
        let slow = Array2::from_elem((24, 24), s0);
        // A skew (knight-ish) direction not exactly representable by axis/diagonal.
        let src = (3, 4);
        let dst = (19, 11);
        let ray = bent_ray_path(&slow, dx, src, dst).expect("ray");
        let euclid = (((dst.0 as f64 - src.0 as f64).powi(2)
            + (dst.1 as f64 - src.1 as f64).powi(2))
        .sqrt())
            * dx;
        let straight_tt = s0 * euclid;
        assert!(
            ray.traveltime >= straight_tt * (1.0 - 1e-12),
            "graph geodesic must be ≥ straight line: {:.6e} < {:.6e}",
            ray.traveltime,
            straight_tt
        );
        // 8-connectivity worst-case overestimate is ≤ √2 ≈ 1.414; in practice
        // mixing axis+diagonal steps keeps it well under ~9%.
        assert!(
            ray.traveltime <= straight_tt * 1.10,
            "8-conn overestimate too large: {:.6e} > 1.10·{:.6e}",
            ray.traveltime,
            straight_tt
        );
    }

    /// Fermat / refraction: a fast (low-slowness) channel off the straight line
    /// lowers the traveltime below the straight-line value and the path enters
    /// the channel — the defining bent-ray behaviour.
    #[test]
    fn fermat_fast_channel_bends_ray_and_lowers_traveltime() {
        let dx = 1e-3;
        let nx = 40;
        let ny = 40;
        let s_bg = 1.0 / 1500.0;
        let s_fast = 1.0 / 3000.0; // 2× faster channel
        let mut slow = Array2::from_elem((nx, ny), s_bg);
        // Horizontal fast channel along row 8 (off the straight chord from
        // (4,5) to (4,34), which runs along row 4).
        for j in 0..ny {
            slow[[8, j]] = s_fast;
        }
        let src = (4, 5);
        let dst = (4, 34);

        // Straight-line (background-only) reference traveltime along the chord.
        let straight_tt = s_bg * (dst.1 - src.1) as f64 * dx;

        let ray = bent_ray_path(&slow, dx, src, dst).expect("ray");

        // (a) The Fermat path is strictly faster than the straight chord.
        assert!(
            ray.traveltime < straight_tt,
            "fast channel must lower traveltime: bent {:.6e} ≥ straight {:.6e}",
            ray.traveltime,
            straight_tt
        );
        // (b) The ray actually detours into the fast channel (reaches row 8).
        let max_row = ray.path.iter().map(|&(i, _)| i).max().unwrap_or(0);
        assert!(
            max_row >= 8,
            "ray must bend into the fast channel (row 8); max row reached = {max_row}"
        );
    }

    /// Endpoint and bounds handling.
    #[test]
    fn degenerate_and_out_of_bounds() {
        let slow = Array2::from_elem((8, 8), 1.0 / 1500.0);
        // Same source and receiver → zero traveltime.
        let r = bent_ray_path(&slow, 1e-3, (3, 3), (3, 3)).expect("self ray");
        assert_eq!(r.traveltime, 0.0);
        // Out-of-bounds receiver → None.
        assert!(bent_ray_path(&slow, 1e-3, (0, 0), (8, 0)).is_none());
    }
}
