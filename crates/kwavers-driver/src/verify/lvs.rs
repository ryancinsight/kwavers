//! LVS — layout versus schematic.
use crate::board::Board;
use std::collections::{HashMap, HashSet};

/// Disjoint-set forest over grid nodes, for as-built connectivity extraction.
struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n as u32).collect(),
        }
    }
    fn find(&mut self, x: u32) -> u32 {
        let mut r = x;
        while self.parent[r as usize] != r {
            r = self.parent[r as usize];
        }
        // Path compression.
        let mut c = x;
        while self.parent[c as usize] != r {
            let next = self.parent[c as usize];
            self.parent[c as usize] = r;
            c = next;
        }
        r
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

/// Layout-versus-schematic findings: as-built copper connectivity compared to the intended netlist.
#[derive(Debug, Clone, Default)]
pub struct LvsReport {
    /// `(net name, island count)` for every intended net whose pads are **not** all on one copper
    /// island — an open (a route the autorouter did not complete, or a missing connection).
    pub opens: Vec<(String, usize)>,
    /// `(net A, net B)` pairs fused onto a single copper island — a short between distinct nets.
    pub shorts: Vec<(String, String)>,
    /// Whether the layout connectivity matches the netlist exactly.
    pub pass: bool,
}

/// Extract as-built electrical connectivity from the routed copper and compare it to the intended
/// netlist. Tracks union the grid cells they cover, vias union layers at a cell, multi-layer pads
/// union their layers, and a copper **pour** (a [`crate::board::Zone`]) ties together all features of
/// its net (a plane connects everything tapped into it). Then:
/// * a net whose pads land on more than one island is an **open**;
/// * an island holding pads of more than one net is a **short**.
#[must_use]
pub fn lvs(board: &Board) -> LvsReport {
    let spec = board.spec;
    let layer_stride = spec.nx * spec.ny;
    let n_nodes = layer_stride * spec.nlayers;
    let mut uf = UnionFind::new(n_nodes);
    let node =
        |ix: usize, iy: usize, layer: usize| (ix + iy * spec.nx + layer * layer_stride) as u32;

    // Tracks: union every consecutive grid cell along the (axis-aligned) segment, so a branch meeting
    // a trunk mid-run is connected, not just the two endpoints.
    for t in &board.tracks {
        let (sx, sy) = spec.cell_of(t.start);
        let (ex, ey) = spec.cell_of(t.end);
        let layer = t.layer.0 as usize;
        if sx == ex {
            let (lo, hi) = (sy.min(ey), sy.max(ey));
            for y in lo..hi {
                uf.union(node(sx, y, layer), node(sx, y + 1, layer));
            }
        } else if sy == ey {
            let (lo, hi) = (sx.min(ex), sx.max(ex));
            for x in lo..hi {
                uf.union(node(x, sy, layer), node(x + 1, sy, layer));
            }
        } else {
            // Diagonal track: union each cell the 45° wire passes through — the on-diagonal cells
            // `(sx,sy), (sx±1,sy±1), …, (ex,ey)`. Unioning the full run (not just the endpoints)
            // keeps a branch that meets the diagonal at an intermediate grid vertex connected — the
            // case `merge_collinear` creates when it fuses several diagonal segments through such a
            // vertex into one long track (the endpoint-only union dropped that vertex and reported a
            // false open). Only the on-diagonal cells are unioned; the off-diagonal staircase corners
            // the wire physically clears are never touched, so this introduces no false shorts (a
            // foreign net actually sitting on an on-diagonal cell *is* a real short kicad flags too).
            let dx: i64 = if ex >= sx { 1 } else { -1 };
            let dy: i64 = if ey >= sy { 1 } else { -1 };
            let nx = (ex as i64 - sx as i64).unsigned_abs();
            let ny = (ey as i64 - sy as i64).unsigned_abs();
            if nx == ny {
                let (mut cx, mut cy) = (sx as i64, sy as i64);
                for _ in 0..nx {
                    uf.union(
                        node(cx as usize, cy as usize, layer),
                        node((cx + dx) as usize, (cy + dy) as usize, layer),
                    );
                    cx += dx;
                    cy += dy;
                }
            } else {
                // Non-45° (shouldn't occur from this router) — endpoint union is the safe fallback.
                uf.union(node(sx, sy, layer), node(ex, ey, layer));
            }
        }
    }
    // Vias: union the layer span at the via cell.
    for v in &board.vias {
        let (vx, vy) = spec.cell_of(v.pos);
        let (lo, hi) = (v.from.0.min(v.to.0) as usize, v.from.0.max(v.to.0) as usize);
        for l in lo..hi {
            uf.union(node(vx, vy, l), node(vx, vy, l + 1));
        }
    }
    // Pads: union the pad cell across the layers the pad's copper occupies. A **thru-hole** pad is a
    // full-stack barrel (its `layers` list — `[0,1]` — is only a multi-layer marker), so it unions
    // *every* copper layer at its cell; an SMD pad unions only the layer it sits on. Without the
    // full-barrel union, a track crossing a thru-hole pad on an inner layer would not be detected as
    // touching that pad's net — the exact short kicad-cli catches but the internal check used to miss.
    for p in &board.pads {
        let (px, py) = spec.cell_of(p.pos);
        let layers: Vec<usize> = if p.layers.len() > 1 {
            (0..spec.nlayers).collect()
        } else {
            p.layers.iter().map(|l| l.0 as usize).collect()
        };
        if let Some((&first, rest)) = layers.split_first() {
            for &l in rest {
                uf.union(node(px, py, first), node(px, py, l));
            }
        }
    }
    // Plane pours: a zone connects every feature of its net (the pour is one continuous copper).
    // Tie all of that net's pads to a single representative so plane-fed pads are not false opens.
    if !board.zones.is_empty() {
        let zone_nets: std::collections::HashSet<u32> =
            board.zones.iter().map(|z| z.net.0).collect();
        let mut rep: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for p in &board.pads {
            if let Some(n) = p.net {
                if zone_nets.contains(&n.0) {
                    let (px, py) = spec.cell_of(p.pos);
                    let layer = p.layers.first().map_or(0, |l| l.0 as usize);
                    let pad_node = node(px, py, layer);
                    match rep.get(&n.0) {
                        Some(&first) => uf.union(first, pad_node),
                        None => {
                            rep.insert(n.0, pad_node);
                        }
                    }
                }
            }
        }
    }

    // Geometric touch-union: two same-net tracks on the same layer whose copper physically overlaps
    // are one electrical island even when they share no grid cell. The chamfer/miter DFM passes leave
    // parallel diagonal segments ~0.14 mm apart — their 0.15 mm-wide copper overlaps (negative edge
    // gap) yet `cell_of` rounds them to distinct cells, so the cell union alone reports a false open
    // that kicad-cli (exact geometry) does not. Grouped by `(net, layer)` so the comparison is the
    // small per-net set, not all-pairs. Same-net only ⇒ this can only *merge* islands, never fabricate
    // a short.
    let mut by_net_layer: HashMap<(u32, u16), Vec<usize>> = HashMap::new();
    for (i, t) in board.tracks.iter().enumerate() {
        by_net_layer
            .entry((t.net.0, t.layer.0))
            .or_default()
            .push(i);
    }
    for idxs in by_net_layer.values() {
        for a in 0..idxs.len() {
            let ta = &board.tracks[idxs[a]];
            let touch_a = ta.width.0 as f64 / 2.0;
            let (ax, ay) = spec.cell_of(ta.start);
            let layer = ta.layer.0 as usize;
            for &j in &idxs[a + 1..] {
                let tb = &board.tracks[j];
                let touch = touch_a + tb.width.0 as f64 / 2.0;
                if crate::geom::dist_seg_seg(ta.start, ta.end, tb.start, tb.end) <= touch {
                    let (bx, by) = spec.cell_of(tb.start);
                    uf.union(node(ax, ay, layer), node(bx, by, layer));
                }
            }
        }
    }

    // Map each netted pad to its copper island, then check opens (per net) and shorts (per island).
    let mut net_islands: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut island_nets: HashMap<u32, HashSet<u32>> = HashMap::new();
    for p in &board.pads {
        if let Some(n) = p.net {
            let (px, py) = spec.cell_of(p.pos);
            let layer = p.layers.first().map_or(0, |l| l.0 as usize);
            let root = uf.find(node(px, py, layer));
            net_islands.entry(n.0).or_default().insert(root);
            island_nets.entry(root).or_default().insert(n.0);
        }
    }
    let mut r = LvsReport::default();
    for (net, islands) in &net_islands {
        if islands.len() > 1 {
            r.opens
                .push((board.nets[*net as usize].name.clone(), islands.len()));
        }
    }
    let mut seen_short: HashSet<(u32, u32)> = HashSet::new();
    for nets in island_nets.values() {
        if nets.len() > 1 {
            let mut v: Vec<u32> = nets.iter().copied().collect();
            v.sort_unstable();
            for i in 0..v.len() {
                for &b in &v[i + 1..] {
                    if seen_short.insert((v[i], b)) {
                        r.shorts.push((
                            board.nets[v[i] as usize].name.clone(),
                            board.nets[b as usize].name.clone(),
                        ));
                    }
                }
            }
        }
    }
    r.opens.sort();
    r.shorts.sort();
    r.pass = r.opens.is_empty() && r.shorts.is_empty();
    r
}
