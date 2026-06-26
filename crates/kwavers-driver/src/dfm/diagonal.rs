//! Diagonal / corner geometry DFM passes: 45° → orthogonal conversion, acid-trap chamfering, via-clearance resolution, and 135° corner mitering.

use crate::board::{Board, Track};
use crate::geom::Point;
use crate::rules::DesignRules;

/// Unconditionally convert every exact-45° diagonal track segment to an orthogonal L-shape
/// (horizontal leg then vertical leg).
///
/// When `diagonal_routing` is enabled, the router emits exact-45° edges (|dx| = |dy|) to
/// shorten paths through congested BGA regions. These segments can create acute-angle junctions
/// (acid traps) at Steiner branch points where a diagonal arm fans out alongside an axial arm
/// going in a similar direction. Rather than requiring multiple selective-repair iterations,
/// converting EVERY diagonal unconditionally guarantees the emitted copper contains no
/// diagonal segments and no acid traps, while preserving the superior routing-path topology
/// the diagonal search found (the route still uses the same channels; only the copper shape
/// per segment changes). Call [`crate::dfm::merge_collinear`] afterward to consolidate duplicate axial
/// legs the replacement may introduce.
///
/// Returns the number of diagonal segments replaced (0 if diagonal routing was not used).
pub fn convert_diagonals_to_orthogonal(board: &mut Board) -> usize {
    let mut kept: Vec<Track> = Vec::with_capacity(board.tracks.len());
    let mut replacements: Vec<Track> = Vec::new();
    let mut count = 0usize;

    for t in board.tracks.drain(..) {
        let dx = (t.end.x.0 - t.start.x.0).abs();
        let dy = (t.end.y.0 - t.start.y.0).abs();
        if dx != 0 && dx == dy {
            // Exact 45° diagonal: replace with L-shape (horizontal then vertical).
            let corner = Point::new(t.end.x, t.start.y);
            if corner != t.start {
                replacements.push(Track {
                    start: t.start,
                    end: corner,
                    width: t.width,
                    layer: t.layer,
                    net: t.net,
                });
            }
            if corner != t.end {
                replacements.push(Track {
                    start: corner,
                    end: t.end,
                    width: t.width,
                    layer: t.layer,
                    net: t.net,
                });
            }
            count += 1;
        } else {
            kept.push(t);
        }
    }
    board.tracks = kept;
    board.tracks.extend(replacements);
    count
}

/// Convert exact-45° diagonal track segments to orthogonal L-shapes, choosing a corner that
/// does not introduce a grid-level short.
///
/// Unlike [`convert_diagonals_to_orthogonal`] (which always picks the horizontal-first corner
/// and can land on cells owned by foreign nets), this variant:
/// 1. Builds a **complete grid-level occupancy map** matching the LVS union rules:
///    - All consecutive grid cells along each axial track segment (not just endpoints), because
///      an L-corner landing mid-track creates the same LVS short as one landing on an endpoint.
///    - All via cells across the via's layer span, because the LVS barrel-unions all layers at
///      each via, so an L-corner on any of those layers is electrically connected to the via's net.
///    - All pad cells across their accessible layers (through-hole pads cover all layers;
///      SMD pads cover only their declared layer).
/// 2. For each diagonal, tries the horizontal-first corner `(end.x, start.y)`; if that cell
///    belongs to a different net on the diagonal's layer, tries the vertical-first corner
///    `(start.x, end.y)`; if both corners are foreign-owned, keeps the diagonal unchanged.
/// 3. Updates the cell map as it goes so that successive conversions see newly claimed corners.
///
/// Returns the number of diagonal segments converted. Call [`crate::dfm::merge_collinear`] afterwards to
/// fold any duplicate axial legs the conversion may produce.
pub fn convert_diagonals_to_orthogonal_safe(board: &mut Board) -> usize {
    use std::collections::HashMap;
    let spec = board.spec;

    // Build a complete grid-level occupancy map: every cell that the LVS considers "owned" by a
    // net must block an L-corner from landing there. The LVS unions:
    //   1. All consecutive grid cells along each axial track segment (not just endpoints).
    //   2. All via cells across the via's layer span.
    //   3. All pad cells across their accessible layers.
    // Seeding only endpoints (the previous approach) missed mid-track cells and via cells, so
    // an L-corner could land on a cell that was mid-track or at a via for a foreign net, which
    // the LVS barrel-union then connected to that net — producing the DOUT/TMS short.
    let mut cell_net: HashMap<(usize, usize, u16), u32> = HashMap::new();

    // Axial tracks: seed every grid cell along the run (mirrors the LVS consecutive-cell union).
    // Diagonal tracks: seed only their two endpoints (no staircase cells have physical copper).
    for t in &board.tracks {
        let (sx, sy) = spec.cell_of(t.start);
        let (ex, ey) = spec.cell_of(t.end);
        let layer = t.layer.0;
        let net = t.net.0;
        if sx == ex {
            // Vertical segment: seed the full column span.
            let (lo, hi) = (sy.min(ey), sy.max(ey));
            for y in lo..=hi {
                cell_net.entry((sx, y, layer)).or_insert(net);
            }
        } else if sy == ey {
            // Horizontal segment: seed the full row span.
            let (lo, hi) = (sx.min(ex), sx.max(ex));
            for x in lo..=hi {
                cell_net.entry((x, sy, layer)).or_insert(net);
            }
        } else {
            // Diagonal: only endpoints have physical copper.
            cell_net.entry((sx, sy, layer)).or_insert(net);
            cell_net.entry((ex, ey, layer)).or_insert(net);
        }
    }

    // Vias: seed all layers in the via's barrel span (the LVS unions the full span at each cell).
    for v in &board.vias {
        let (vx, vy) = spec.cell_of(v.pos);
        let lo = v.from.0.min(v.to.0);
        let hi = v.from.0.max(v.to.0);
        for l in lo..=hi {
            cell_net.entry((vx, vy, l)).or_insert(v.net.0);
        }
    }

    // Pads: through-hole pads are full-stack barrels (the LVS unions all layers at their cell);
    // SMD pads occupy only their declared copper layer.
    for p in &board.pads {
        if let Some(n) = p.net {
            let (px, py) = spec.cell_of(p.pos);
            if p.layers.len() > 1 {
                for l in 0..(spec.nlayers as u16) {
                    cell_net.entry((px, py, l)).or_insert(n.0);
                }
            } else {
                for layer_id in &p.layers {
                    cell_net.entry((px, py, layer_id.0)).or_insert(n.0);
                }
            }
        }
    }

    let mut kept: Vec<Track> = Vec::with_capacity(board.tracks.len());
    let mut replacements: Vec<Track> = Vec::new();
    let mut count = 0usize;

    for t in board.tracks.drain(..) {
        let dx = (t.end.x.0 - t.start.x.0).abs();
        let dy = (t.end.y.0 - t.start.y.0).abs();
        if dx == 0 || dx != dy {
            kept.push(t);
            continue;
        }

        let (sx, sy) = spec.cell_of(t.start);
        let (ex, ey) = spec.cell_of(t.end);
        let layer = t.layer.0;
        let net = t.net.0;

        // Try horizontal-first corner (end.x, start.y).
        let c1_safe = cell_net.get(&(ex, sy, layer)).map_or(true, |&n| n == net);
        // Try vertical-first corner (start.x, end.y).
        let c2_safe = cell_net.get(&(sx, ey, layer)).map_or(true, |&n| n == net);

        let corner = if c1_safe {
            cell_net.insert((ex, sy, layer), net);
            Point::new(t.end.x, t.start.y)
        } else if c2_safe {
            cell_net.insert((sx, ey, layer), net);
            Point::new(t.start.x, t.end.y)
        } else {
            // Both corners are claimed by foreign nets — keep as diagonal.
            kept.push(t);
            continue;
        };

        if corner != t.start {
            replacements.push(Track {
                start: t.start,
                end: corner,
                width: t.width,
                layer: t.layer,
                net: t.net,
            });
        }
        if corner != t.end {
            replacements.push(Track {
                start: corner,
                end: t.end,
                width: t.width,
                layer: t.layer,
                net: t.net,
            });
        }
        count += 1;
    }

    board.tracks = kept;
    board.tracks.extend(replacements);
    count
}

/// diagonal segment with a two-leg orthogonal L-shape selected from the offending junction.
///
/// A 45° diagonal meeting an axial segment at a < 90° interior angle traps etchant in the narrow
/// pocket (acid trap) — a DFM reject. Pure Manhattan routing never produces this; diagonal routing
/// can when the search exits a diagonal run onto an axial that heads in a "similar" direction at a
/// Steiner branch or at a turn. The L-shape replacement maintains connectivity and copper area
/// while guaranteeing every junction is ≥ 90°. After this pass, call [`crate::dfm::merge_collinear`] again to
/// fold any duplicate axial segments the replacement may create (e.g., when the L-corner lands on
/// the endpoint of an adjacent axial segment going the same direction).
///
/// Returns the number of diagonal segments replaced (0 if no acid traps are present).
pub fn chamfer_diagonal_traps(board: &mut Board) -> usize {
    let key = |p: Point| (p.x.0, p.y.0);

    fn chamfer_corner(track: &Track, apex: Point, other: Point) -> Point {
        let vx = other.x.0 - apex.x.0;
        let vy = other.y.0 - apex.y.0;
        if apex == track.start {
            if vy == 0 {
                Point::new(track.start.x, track.end.y)
            } else {
                Point::new(track.end.x, track.start.y)
            }
        } else if vx == 0 {
            Point::new(track.start.x, track.end.y)
        } else {
            Point::new(track.end.x, track.start.y)
        }
    }

    // --- Step 1: identify diagonal tracks that participate in an acute-angle junction. ---
    let mut marked: std::collections::HashMap<usize, Point> = std::collections::HashMap::new();
    let tr = &board.tracks;

    for i in 0..tr.len() {
        let dxi = (tr[i].end.x.0 - tr[i].start.x.0).abs();
        let dyi = (tr[i].end.y.0 - tr[i].start.y.0).abs();
        // Only test diagonal tracks (|dx| == |dy| != 0).
        if dxi == 0 || dxi != dyi {
            continue;
        }
        for j in (i + 1)..tr.len() {
            if tr[i].net != tr[j].net || tr[i].layer != tr[j].layer {
                continue;
            }
            let (a, b) = (tr[i].start, tr[i].end);
            let (c, d) = (tr[j].start, tr[j].end);
            // Find the shared endpoint; u and v are the opposite arms from the apex.
            let (apex, u, v) = if key(a) == key(c) {
                (a, b, d)
            } else if key(a) == key(d) {
                (a, b, c)
            } else if key(b) == key(c) {
                (b, a, d)
            } else if key(b) == key(d) {
                (b, a, c)
            } else {
                continue;
            };
            let ux = (u.x.0 - apex.x.0) as f64;
            let uy = (u.y.0 - apex.y.0) as f64;
            let vx = (v.x.0 - apex.x.0) as f64;
            let vy = (v.y.0 - apex.y.0) as f64;
            if ux * vx + uy * vy > 0.0 {
                // Acute junction: mark track i (the diagonal) for replacement. The replacement
                // corner is selected so the leg leaving the acute apex is perpendicular to the
                // other same-net branch at that apex.
                marked
                    .entry(i)
                    .or_insert_with(|| chamfer_corner(&tr[i], apex, v));
                // If j is also diagonal, mark it too — defensive, catches multi-diagonal branches.
                let dxj = (tr[j].end.x.0 - tr[j].start.x.0).abs();
                let dyj = (tr[j].end.y.0 - tr[j].start.y.0).abs();
                if dxj != 0 && dxj == dyj {
                    marked
                        .entry(j)
                        .or_insert_with(|| chamfer_corner(&tr[j], apex, u));
                }
            }
        }
    }

    if marked.is_empty() {
        return 0;
    }
    let count = marked.len();

    // --- Step 2: replace each marked diagonal with a horizontal + vertical L-shape. ---
    let mut replacements: Vec<Track> = Vec::with_capacity(count * 2);
    let mut kept: Vec<Track> = Vec::with_capacity(board.tracks.len() - count);

    for (i, t) in board.tracks.drain(..).enumerate() {
        let Some(corner) = marked.get(&i).copied() else {
            kept.push(t);
            continue;
        };
        if corner != t.start {
            replacements.push(Track {
                start: t.start,
                end: corner,
                width: t.width,
                layer: t.layer,
                net: t.net,
            });
        }
        if corner != t.end {
            replacements.push(Track {
                start: corner,
                end: t.end,
                width: t.width,
                layer: t.layer,
                net: t.net,
            });
        }
    }

    board.tracks = kept;
    board.tracks.extend(replacements);
    count
}

/// Detect and repair diagonal track segments that violate geometric copper clearance to
/// foreign-net vias.
///
/// KiCad DRC checks the true Euclidean edge-to-edge gap between a diagonal track's copper hull
/// and a via's annular ring, which can be below `min_clearance` even when the router's grid-level
/// corner-cell guards are all clear. This function measures the perpendicular distance from each
/// foreign-net via centre to each diagonal's centre-line; if the resulting edge-to-edge gap is
/// below `rules.min_clearance`, it converts the diagonal to an orthogonal L-shape by selecting
/// the horizontal-first or vertical-first corner that maximises clearance. If neither corner
/// achieves the required clearance, the diagonal is left unchanged (best-effort).
///
/// Returns the number of diagonal segments converted.
pub fn resolve_diagonal_via_clearance(board: &mut Board, rules: &DesignRules) -> usize {
    use crate::geom::dist_point_seg;
    let mc = rules.min_clearance.0 as f64; // nm

    // Build a per-layer list of (via_pos, outer_radius_nm, via_net).
    struct ViaRec {
        pos: Point,
        outer_radius: f64, // nm
        net: u32,
    }
    let mut vias_by_layer: std::collections::HashMap<u16, Vec<ViaRec>> =
        std::collections::HashMap::new();
    for v in &board.vias {
        let lo = v.from.0.min(v.to.0);
        let hi = v.from.0.max(v.to.0);
        let r = v.diameter.0 as f64 / 2.0;
        for l in lo..=hi {
            vias_by_layer.entry(l).or_default().push(ViaRec {
                pos: v.pos,
                outer_radius: r,
                net: v.net.0,
            });
        }
    }

    if vias_by_layer.is_empty() {
        return 0;
    }

    // Identify violating diagonals.
    let mut violating: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, t) in board.tracks.iter().enumerate() {
        let dx = (t.end.x.0 - t.start.x.0).abs();
        let dy = (t.end.y.0 - t.start.y.0).abs();
        if dx == 0 || dx != dy {
            continue;
        }
        let layer = t.layer.0;
        let Some(vias) = vias_by_layer.get(&layer) else {
            continue;
        };
        let half = t.width.0 as f64 / 2.0;
        if vias.iter().any(|via| {
            if via.net == t.net.0 {
                return false;
            }
            let d = dist_point_seg(via.pos, t.start, t.end) - half - via.outer_radius;
            d < mc
        }) {
            violating.insert(i);
        }
    }

    if violating.is_empty() {
        return 0;
    }

    let mut kept: Vec<Track> = Vec::with_capacity(board.tracks.len());
    let mut replacements: Vec<Track> = Vec::new();
    let mut count = 0usize;

    for (i, t) in board.tracks.drain(..).enumerate() {
        if !violating.contains(&i) {
            kept.push(t);
            continue;
        }

        let half = t.width.0 as f64 / 2.0;
        let layer = t.layer.0;
        // Extract Copy fields so the clearance closure does not borrow `t` as a whole.
        let t_net_0 = t.net.0;
        let t_start = t.start;
        let t_end = t.end;

        // Try horizontal-first corner (end.x, start.y), then vertical-first (start.x, end.y).
        let c1 = Point::new(t_end.x, t_start.y);
        let c2 = Point::new(t_start.x, t_end.y);

        // Build a net-filtered clearance check for the two corners.
        let ok = |c: Point| -> bool {
            let Some(vias) = vias_by_layer.get(&layer) else {
                return true;
            };
            vias.iter().all(|via| {
                if via.net == t_net_0 {
                    return true;
                }
                let d1 = dist_point_seg(via.pos, t_start, c) - half - via.outer_radius;
                let d2 = dist_point_seg(via.pos, c, t_end) - half - via.outer_radius;
                d1 >= mc && d2 >= mc
            })
        };

        let corner = if ok(c1) {
            c1
        } else if ok(c2) {
            c2
        } else {
            kept.push(t);
            continue;
        };

        if corner != t.start {
            replacements.push(Track {
                start: t.start,
                end: corner,
                width: t.width,
                layer: t.layer,
                net: t.net,
            });
        }
        if corner != t.end {
            replacements.push(Track {
                start: corner,
                end: t.end,
                width: t.width,
                layer: t.layer,
                net: t.net,
            });
        }
        count += 1;
    }

    board.tracks = kept;
    board.tracks.extend(replacements);
    count
}
