//! Track-geometry DFM passes: collinear merge, pad-entry stubs, body-junction splits, and dangling-stub + orphan-copper cleanup.

use crate::board::{Board, LayerId, NetId, Track};
use crate::geom::{dist_point_seg, orient, Nm, Point};
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::rules::DesignRules;

/// Merge runs of collinear, same-`(net, layer, width)` axis-aligned track segments into one segment
/// each. Copper is the set union of the original intervals — identical area, fewer vertices. Returns
/// the number of segments removed (the consolidation that occurred).
///
/// Non-axis-aligned or zero-length segments are passed through untouched (the router is Manhattan, so
/// this is just a safety net). Output order is deterministic (sorted group key, then interval).
pub fn merge_collinear(board: &mut Board) -> usize {
    use std::collections::HashMap;
    // Group key: (net, layer, width, orientation [0 = horizontal, 1 = vertical], fixed coordinate).
    // Value: the [lo, hi] extents along the free axis for each segment in the group.
    type Key = (u32, u16, i64, u8, i64);
    let mut groups: HashMap<Key, Vec<(i64, i64)>> = HashMap::new();
    let mut out: Vec<Track> = Vec::new();
    let before = board.tracks.len();

    for t in board.tracks.drain(..) {
        let horiz = t.start.y.0 == t.end.y.0 && t.start.x.0 != t.end.x.0;
        let vert = t.start.x.0 == t.end.x.0 && t.start.y.0 != t.end.y.0;
        if horiz {
            let (lo, hi) = (t.start.x.0.min(t.end.x.0), t.start.x.0.max(t.end.x.0));
            groups
                .entry((t.net.0, t.layer.0, t.width.0, 0, t.start.y.0))
                .or_default()
                .push((lo, hi));
        } else if vert {
            let (lo, hi) = (t.start.y.0.min(t.end.y.0), t.start.y.0.max(t.end.y.0));
            groups
                .entry((t.net.0, t.layer.0, t.width.0, 1, t.start.x.0))
                .or_default()
                .push((lo, hi));
        } else {
            out.push(t); // diagonal / zero-length: leave as is
        }
    }

    let mut keys: Vec<Key> = groups.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        let mut iv = groups.remove(&k).expect("key came from the map");
        iv.sort_unstable();
        // Sweep-merge touching/overlapping intervals (abutting counts: lo <= cur.hi).
        let mut cur = iv[0];
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for &(lo, hi) in &iv[1..] {
            if lo <= cur.1 {
                cur.1 = cur.1.max(hi);
            } else {
                merged.push(cur);
                cur = (lo, hi);
            }
        }
        merged.push(cur);

        let (net, layer, width, orient, fixed) = k;
        for (lo, hi) in merged {
            let (start, end) = if orient == 0 {
                (Point::new(Nm(lo), Nm(fixed)), Point::new(Nm(hi), Nm(fixed)))
            } else {
                (Point::new(Nm(fixed), Nm(lo)), Point::new(Nm(fixed), Nm(hi)))
            };
            out.push(Track {
                start,
                end,
                width: Nm(width),
                layer: LayerId(layer),
                net: NetId(net),
            });
        }
    }

    board.tracks = out;
    before.saturating_sub(board.tracks.len())
}

/// Add short same-net entry stubs from each pad's exact centre to the grid node the router targets.
///
/// The router works on grid-cell centres. Internal LVS can treat a pad and a route in the same cell
/// as connected, but KiCad DRC uses exact continuous geometry. This pass makes that connection
/// explicit as copper in the board model. For a drilled pad, the plated barrel is accessible from
/// every copper layer, so a short stub is emitted on every layer; for SMD pads, only their declared
/// copper layers are used. Returns the number of stubs added.
pub(crate) fn pad_entry_stubs(
    board: &mut Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> usize {
    let spec = board.spec;

    // Pre-build the full board pad table for cross-component clearance checking.
    // Each entry: (pad centre, half-diagonal in Nm units, net-or-None).
    // Using f64 throughout to match dist_point_seg's coordinate space.
    let all_pads: Vec<(crate::geom::Point, f64, Option<crate::board::NetId>)> = comps
        .iter()
        .flat_map(|oc| {
            let ofp = &lib[oc.fp];
            ofp.pads.iter().enumerate().map(move |(ok, opad)| {
                let opos = oc.pad_pos(lib, ok);
                let (opw, oph) = oc.placement.rot.apply_size(opad.size);
                let half_diag = (opw.0 as f64).hypot(oph.0 as f64) / 2.0;
                (opos, half_diag, oc.nets[ok])
            })
        })
        .collect();

    let mut added = 0usize;
    for c in comps {
        let fp = &lib[c.fp];
        for (k, pad) in fp.pads.iter().enumerate() {
            let Some(net) = c.nets[k] else { continue };
            let exact = c.pad_pos(lib, k);
            let (ix, iy) = spec.cell_of(exact);
            let snapped = spec.point_of(ix, iy);
            if snapped == exact {
                continue;
            }
            // Guard: skip if the stub [snapped→exact] would violate min_clearance against
            // ANY pad on the board with a different net (including unconnected pads with None).
            // Using half-diagonal of each obstacle pad as a conservative copper-edge radius.
            let class = board.class_of(net);
            let track_width = rules.track_for(class);
            let guard_margin = rules.min_clearance.0 as f64 + (track_width.0 / 2) as f64;
            let would_violate = all_pads.iter().any(|&(opos, half_diag, onet)| {
                if onet == Some(net) {
                    return false; // same net: stub may overlap freely
                }
                if opos == exact {
                    return false; // this IS the same pad (shouldn't happen, belt-and-suspenders)
                }
                let min_center_dist = half_diag + guard_margin;
                dist_point_seg(opos, snapped, exact) < min_center_dist
            });
            if would_violate {
                continue;
            }
            // Thru-hole pads get a stub on every layer; SMD pads use their single layer.
            // No intermediate Vec: push inline in each branch to avoid the heap allocation.
            if pad.layers.len() > 1 {
                for l in 0..spec.nlayers {
                    board.tracks.push(Track {
                        start: snapped,
                        end: exact,
                        width: track_width,
                        layer: LayerId(l as u16),
                        net,
                    });
                    added += 1;
                }
            } else {
                for &layer in &pad.layers {
                    board.tracks.push(Track {
                        start: snapped,
                        end: exact,
                        width: track_width,
                        layer,
                        net,
                    });
                    added += 1;
                }
            }
        }
    }
    added
}

/// Split a same-net segment whenever another same-net track endpoint lands on its interior.
///
/// KiCad DRC treats an endpoint touching a segment body as an unsplit junction for manufacturing
/// checks such as `track_dangling`. Splitting the body segment at that coordinate preserves copper
/// area exactly while making the electrical/DFM junction explicit in the board model.
pub(crate) fn split_track_body_junctions(board: &mut Board) -> usize {
    let mut split_points: Vec<Vec<Point>> = vec![Vec::new(); board.tracks.len()];
    for (endpoint_idx, endpoint_track) in board.tracks.iter().enumerate() {
        for endpoint in [endpoint_track.start, endpoint_track.end] {
            for (body_idx, body_track) in board.tracks.iter().enumerate() {
                if endpoint_idx == body_idx
                    || endpoint_track.net != body_track.net
                    || endpoint_track.layer != body_track.layer
                    || !point_on_segment_interior(endpoint, body_track.start, body_track.end)
                {
                    continue;
                }
                split_points[body_idx].push(endpoint);
            }
        }
    }

    let mut split_count = 0usize;
    let mut out = Vec::with_capacity(board.tracks.len());
    for (track, mut points) in board.tracks.drain(..).zip(split_points) {
        if points.is_empty() {
            out.push(track);
            continue;
        }
        points.sort_by_key(|p| track_order_key(track.start, track.end, *p));
        points.dedup();

        let mut chain = Vec::with_capacity(points.len() + 2);
        chain.push(track.start);
        chain.extend(points);
        chain.push(track.end);

        for pair in chain.windows(2) {
            let start = pair[0];
            let end = pair[1];
            if start != end {
                out.push(Track {
                    start,
                    end,
                    width: track.width,
                    layer: track.layer,
                    net: track.net,
                });
            }
        }
        split_count += chain.len().saturating_sub(2);
    }
    board.tracks = out;
    split_count
}

fn point_on_segment_interior(p: Point, a: Point, b: Point) -> bool {
    p != a
        && p != b
        && orient(a, b, p) == 0
        && p.x.0 >= a.x.0.min(b.x.0)
        && p.x.0 <= a.x.0.max(b.x.0)
        && p.y.0 >= a.y.0.min(b.y.0)
        && p.y.0 <= a.y.0.max(b.y.0)
}

fn track_order_key(a: Point, b: Point, p: Point) -> i64 {
    let dx = (b.x.0 - a.x.0).abs();
    let dy = (b.y.0 - a.y.0).abs();
    if dx >= dy {
        if b.x.0 >= a.x.0 {
            p.x.0
        } else {
            -p.x.0
        }
    } else if b.y.0 >= a.y.0 {
        p.y.0
    } else {
        -p.y.0
    }
}

/// Remove **dangling track stubs** — antennas. A track endpoint that is the unique end of exactly
/// one segment (degree 1) and lands on no pad, via, or other same-net track is a floating end: it
/// carries no connectivity but acts as an unintended EMI radiator/receiver and is a DFM/signal-
/// integrity defect (an unterminated transmission-line stub). Each pass deletes every segment with
/// such a free end; removing a segment can expose a new degree-1 end one cell back, so the pass is
/// repeated until stable, peeling each stub back to the junction (degree ≥ 3) or anchor that roots
/// it. Connectivity is preserved: a stub by definition reaches nothing, so its removal opens no net.
/// Returns the number of segments removed.
pub fn trim_dangling_stubs(board: &mut Board) -> usize {
    use std::collections::HashMap;
    let tol = board.spec.pitch.0 as f64; // within one cell of a pad/via ⇒ anchored, not dangling
    let key = |p: Point| (p.x.0, p.y.0);
    let anchors: Vec<Point> = board
        .pads
        .iter()
        .map(|p| p.pos)
        .chain(board.vias.iter().map(|v| v.pos))
        .collect();
    let is_anchor = |p: Point| anchors.iter().any(|a| a.euclid(p) <= tol);

    let mut removed_total = 0;
    loop {
        let mut deg: HashMap<(i64, i64), u32> = HashMap::new();
        for t in &board.tracks {
            *deg.entry(key(t.start)).or_default() += 1;
            *deg.entry(key(t.end)).or_default() += 1;
        }
        let before = board.tracks.len();
        board.tracks.retain(|t| {
            let dangling = |p: Point| deg[&key(p)] == 1 && !is_anchor(p);
            !(dangling(t.start) || dangling(t.end))
        });
        let removed = before - board.tracks.len();
        removed_total += removed;
        if removed == 0 {
            break;
        }
    }
    removed_total
}

/// Remove **orphan copper** — tracks and vias that belong to a copper island containing no pad.
///
/// A negotiated re-route that rips up a net and re-routes it can leave a small disconnected fragment
/// behind (a via plus a stub, an unfinished escape) that connects to nothing. Such pad-less copper is
/// pure liability: it is an EMI antenna, it shows up as a `via_dangling` DRC item, and — being copper
/// with a net but no electrical purpose — it can short or mask-bridge a foreign feature it happens to
/// overlap (e.g. a fiducial). Removing it is always connectivity-safe: every net's real connections
/// live on islands that *do* contain its pads, so deleting a pad-less island opens nothing.
///
/// Connectivity mirrors [`crate::verify::lvs()`]: union grid cells along each track and via barrel and
/// across each pad's layers, plus a geometric touch-union of overlapping same-net tracks (so a stub
/// joined to a pad only by overlapping copper is *not* mistaken for an orphan). Returns the number of
/// track + via segments removed.
pub(crate) fn remove_orphan_copper(board: &mut Board) -> usize {
    let spec = board.spec;
    let stride = spec.nx * spec.ny;
    let n = stride * spec.nlayers;
    let node = |ix: usize, iy: usize, l: usize| ix + iy * spec.nx + l * stride;

    // Minimal union-find with path halving.
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    let union = |parent: &mut Vec<u32>, a: usize, b: usize| {
        let (ra, rb) = (find(parent, a as u32), find(parent, b as u32));
        if ra != rb {
            parent[ra as usize] = rb;
        }
    };

    for t in &board.tracks {
        let (sx, sy) = spec.cell_of(t.start);
        let (ex, ey) = spec.cell_of(t.end);
        let l = t.layer.0 as usize;
        // Union *every* cell the track passes through (not just endpoints), so a via or branch meeting
        // the track mid-run is correctly seen as connected — otherwise legitimate copper could be
        // mis-classified as orphan and wrongly deleted. Axial: walk the run; diagonal: the on-diagonal
        // step cells; arbitrary (shouldn't occur): endpoints (safe over-keep).
        if sx == ex {
            for y in sy.min(ey)..sy.max(ey) {
                union(&mut parent, node(sx, y, l), node(sx, y + 1, l));
            }
        } else if sy == ey {
            for x in sx.min(ex)..sx.max(ex) {
                union(&mut parent, node(x, sy, l), node(x + 1, sy, l));
            }
        } else if (ex as i64 - sx as i64).unsigned_abs() == (ey as i64 - sy as i64).unsigned_abs() {
            let dx: i64 = if ex >= sx { 1 } else { -1 };
            let dy: i64 = if ey >= sy { 1 } else { -1 };
            let (mut cx, mut cy) = (sx as i64, sy as i64);
            for _ in 0..(ex as i64 - sx as i64).unsigned_abs() {
                union(
                    &mut parent,
                    node(cx as usize, cy as usize, l),
                    node((cx + dx) as usize, (cy + dy) as usize, l),
                );
                cx += dx;
                cy += dy;
            }
        } else {
            union(&mut parent, node(sx, sy, l), node(ex, ey, l));
        }
    }
    for v in &board.vias {
        let (vx, vy) = spec.cell_of(v.pos);
        let (lo, hi) = (v.from.0.min(v.to.0) as usize, v.from.0.max(v.to.0) as usize);
        for l in lo..hi {
            union(&mut parent, node(vx, vy, l), node(vx, vy, l + 1));
        }
    }
    for p in &board.pads {
        let (px, py) = spec.cell_of(p.pos);
        let layers: Vec<usize> = if p.layers.len() > 1 {
            (0..spec.nlayers).collect()
        } else {
            p.layers.iter().map(|l| l.0 as usize).collect()
        };
        if let Some((&first, rest)) = layers.split_first() {
            for &l in rest {
                union(&mut parent, node(px, py, first), node(px, py, l));
            }
        }
    }
    // Touch-union of overlapping same-net tracks (per net+layer; cheap).
    use std::collections::HashMap;
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
            let (ax, ay) = spec.cell_of(ta.start);
            let l = ta.layer.0 as usize;
            let ha = ta.width.0 as f64 / 2.0;
            for &j in &idxs[a + 1..] {
                let tb = &board.tracks[j];
                if crate::geom::dist_seg_seg(ta.start, ta.end, tb.start, tb.end)
                    <= ha + tb.width.0 as f64 / 2.0
                {
                    let (bx, by) = spec.cell_of(tb.start);
                    union(&mut parent, node(ax, ay, l), node(bx, by, l));
                }
            }
        }
    }

    // Roots that contain at least one *netted* pad are legitimate copper; everything else is orphan.
    // No-net pads (fiducials, NPTH board locks) are mechanical features, not electrical anchors — a
    // GND stub that merely *overlaps* a no-net fiducial is exactly the orphan-that-shorts-a-fiducial
    // case we must delete, so it cannot count as anchored.
    let mut pad_roots = std::collections::HashSet::new();
    for p in &board.pads {
        if p.net.is_none() {
            continue;
        }
        let (px, py) = spec.cell_of(p.pos);
        let l = p.layers.first().map_or(0, |x| x.0 as usize);
        pad_roots.insert(find(&mut parent, node(px, py, l) as u32));
    }

    let before = board.tracks.len() + board.vias.len();
    board.tracks.retain(|t| {
        let (sx, sy) = spec.cell_of(t.start);
        pad_roots.contains(&find(&mut parent, node(sx, sy, t.layer.0 as usize) as u32))
    });
    board.vias.retain(|v| {
        let (vx, vy) = spec.cell_of(v.pos);
        let l = v.from.0.min(v.to.0) as usize;
        pad_roots.contains(&find(&mut parent, node(vx, vy, l) as u32))
    });
    before - (board.tracks.len() + board.vias.len())
}
