//! Copper-area DFM passes: ampacity widening, quietest-layer selection, and the ground-plane pour.

use crate::board::{Board, LayerId, NetId, Track, Zone, ZoneFill};
use crate::geom::{dist_seg_seg, Nm, Point};

/// Widen each routed track toward its **ampacity requirement** wherever clearance allows — a
/// post-route, copper-only DFM pass (the router lays minimum-width tracks; this fattens the
/// current-carrying ones up to what IPC-2221 needs for their net's RMS current, bounded by the room
/// to the nearest foreign copper so design-rule clearance is preserved).
///
/// `current_of(net)` is the net's RMS current (A); `dt_c` the allowed rise; `oz` the copper weight;
/// `min_clearance` the spacing rule; `edge_clearance` the copper-to-board-edge rule. Plane-backed
/// nets (a [`Zone`] on the net) are skipped — their current flows in the pour. A track is only ever
/// widened, never narrowed. Returns the count widened.
pub fn widen_for_ampacity(
    board: &mut Board,
    current_of: impl Fn(NetId) -> f64,
    dt_c: f64,
    oz: f64,
    min_clearance: Nm,
    edge_clearance: Nm,
) -> usize {
    let clr = min_clearance.0 as f64;
    // Conservative foreign-feature radii (board pads carry no size here): a ~0.6 mm pad ⇒ 0.3 mm.
    let pad_r = Nm::from_mm(0.3).0 as f64;
    let plane_nets: std::collections::HashSet<u32> = board.zones.iter().map(|z| z.net.0).collect();
    let mut widened = 0;
    for i in 0..board.tracks.len() {
        let t = board.tracks[i];
        if plane_nets.contains(&t.net.0) {
            continue;
        }
        let i_rms = current_of(t.net);
        if i_rms <= 0.0 {
            continue;
        }
        let target = crate::physics::ampacity::ipc2221_min_width(i_rms, dt_c, oz, true).0 as f64;
        let cur = t.width.0 as f64;
        if target <= cur {
            continue; // already adequate
        }
        // Largest half-width keeping `min_clearance` to every foreign feature on this track's layer.
        // A foreign track's o_target = max(ipc_min_width(o), o.width): once widened, o.width ≥
        // ipc_min_width(o), so o_target is idempotent — iterating board.tracks directly is correct.
        let edge_half = track_edge_half_limit(board.spec, &t, edge_clearance);
        let mut max_half = edge_half;
        for j in 0..board.tracks.len() {
            let o = board.tracks[j];
            if o.net == t.net || o.layer != t.layer {
                continue;
            }
            let o_target = if plane_nets.contains(&o.net.0) {
                o.width.0 as f64
            } else {
                let oi = current_of(o.net);
                (crate::physics::ampacity::ipc2221_min_width(oi.max(0.0), dt_c, oz, true).0 as f64)
                    .max(o.width.0 as f64)
            };
            let d = dist_seg_seg(t.start, t.end, o.start, o.end);
            max_half = max_half.min(d - o_target / 2.0 - clr);
        }
        for v in &board.vias {
            if v.net == t.net {
                continue;
            }
            let d = crate::geom::dist_point_seg(v.pos, t.start, t.end);
            max_half = max_half.min(d - v.diameter.0 as f64 / 2.0 - clr);
        }
        for p in &board.pads {
            if p.net == Some(t.net) {
                continue;
            }
            let d = crate::geom::dist_point_seg(p.pos, t.start, t.end);
            max_half = max_half.min(d - pad_r - clr);
        }
        let new_w = target.min(2.0 * max_half).max(cur);
        if new_w > cur + 1.0 {
            board.tracks[i].width = Nm(new_w as i64);
            widened += 1;
        }
    }
    widened
}

fn track_edge_half_limit(spec: crate::geom::GridSpec, t: &Track, edge_clearance: Nm) -> f64 {
    let board_min_x = spec.origin.x.0;
    let board_min_y = spec.origin.y.0;
    let board_max_x = spec.origin.x.0 + (spec.nx as i64 - 1) * spec.pitch.0;
    let board_max_y = spec.origin.y.0 + (spec.ny as i64 - 1) * spec.pitch.0;
    [
        t.start.x.0.min(t.end.x.0) - board_min_x,
        t.start.y.0.min(t.end.y.0) - board_min_y,
        board_max_x - t.start.x.0.max(t.end.x.0),
        board_max_y - t.start.y.0.max(t.end.y.0),
    ]
    .into_iter()
    .min()
    .unwrap_or(i64::MAX) as f64
        - edge_clearance.0 as f64
}

/// The copper layer carrying the least routed track length — the best host for a ground plane,
/// since the fewer signal tracks carve it, the larger and more connected the resulting plane.
#[must_use]
pub fn quietest_layer(board: &Board) -> LayerId {
    let mut len = vec![0.0f64; board.spec.nlayers];
    for t in &board.tracks {
        let l = t.layer.0 as usize;
        if l < len.len() {
            len[l] += t.start.euclid(t.end);
        }
    }
    let l = len
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    LayerId(l as u16)
}

/// Pour a `ground`-net plane on a single `layer`, bounded by the **convex hull of the ground
/// features** (GND pads and vias) clipped to the board minus `inset`. KiCad's filler carves
/// design-rule clearance around all foreign copper, so this is clearance-safe; it provides an EMI
/// return-current reference plane and balances copper (reflow warpage). Bounding the pour to the
/// ground-populated hull — rather than flooding the whole board — keeps every plane region near a
/// ground connection, so no far-edge region fills without a tie-in and then reads as isolated
/// copper. The pour goes on the **quietest** layer ([`quietest_layer`]) so few signal tracks carve
/// it. Falls back to the inset board rectangle if there are too few ground features for a hull.
/// Returns `true` (one zone added).
pub fn ground_pour(board: &mut Board, ground: NetId, inset: Nm, layer: LayerId) -> bool {
    let spec = board.spec;
    let w = (spec.nx as i64 - 1) * spec.pitch.0;
    let h = (spec.ny as i64 - 1) * spec.pitch.0;
    let i = inset.0;
    let clamp = |p: Point| Point::new(Nm(p.x.0.clamp(i, w - i)), Nm(p.y.0.clamp(i, h - i)));

    let is_inner = layer.0 > 0 && (layer.0 as usize) < board.spec.nlayers - 1;

    let polygon: Vec<Point> = if is_inner {
        vec![
            Point::new(Nm(i), Nm(i)),
            Point::new(Nm(w - i), Nm(i)),
            Point::new(Nm(w - i), Nm(h - i)),
            Point::new(Nm(i), Nm(h - i)),
        ]
    } else {
        // Ground features: GND pads and GND vias. Their convex hull is the region the plane should
        // cover; beyond it there is no ground to tie a fill to.
        let mut feats: Vec<Point> = board
            .pads
            .iter()
            .filter(|p| p.net == Some(ground))
            .map(|p| p.pos)
            .collect();
        feats.extend(board.vias.iter().filter(|v| v.net == ground).map(|v| v.pos));

        let hull = crate::geom::convex_hull(&feats);
        if hull.len() >= 3 {
            // Inflate the hull outward from its centroid by a few mm so ground features ON the hull
            // boundary (e.g. an edge connector's GND pads) end up *surrounded* by plane — otherwise a
            // boundary pad's thermal relief cannot form the required spokes (`starved_thermal`). The
            // inflation is small and bounded, so the plane still hugs the ground-populated region.
            let n = hull.len() as f64;
            let cx = hull.iter().map(|p| p.x.0 as f64).sum::<f64>() / n;
            let cy = hull.iter().map(|p| p.y.0 as f64).sum::<f64>() / n;
            let grow = Nm::from_mm(3.0).0 as f64;
            hull.into_iter()
                .map(|p| {
                    let (dx, dy) = (p.x.0 as f64 - cx, p.y.0 as f64 - cy);
                    let d = (dx * dx + dy * dy).sqrt().max(1.0);
                    clamp(Point::new(
                        Nm((p.x.0 as f64 + dx / d * grow) as i64),
                        Nm((p.y.0 as f64 + dy / d * grow) as i64),
                    ))
                })
                .collect()
        } else {
            vec![
                Point::new(Nm(i), Nm(i)),
                Point::new(Nm(w - i), Nm(i)),
                Point::new(Nm(w - i), Nm(h - i)),
                Point::new(Nm(i), Nm(h - i)),
            ]
        }
    };
    board.zones.push(Zone {
        net: ground,
        layer,
        polygon,
        fill: ZoneFill::ThermalRelief,
    });
    true
}
