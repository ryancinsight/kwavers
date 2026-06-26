//! Via DFM passes: coincident-via dedup, teardrops, and plane-net via distribution.

use crate::board::{Board, LayerId, NetId, Zone, ZoneFill};
use crate::geom::{Nm, Point};
use crate::rules::DesignRules;

/// Merge vias that share a net and a position (an inner-layer BGA fanout via coinciding with the
/// router's own via at the same ball cell) into one via spanning the union of their layer ranges —
/// otherwise two stacked vias at the same hole read as a co-located drill fault. Returns the count
/// removed.
pub fn dedup_vias(board: &mut Board, rules: &DesignRules) -> usize {
    use std::collections::HashMap;
    // Per coincident (x, y, net): accumulated (lo layer, hi layer, any-leg-filled, representative via).
    type Merge = (u16, u16, bool, crate::board::Via);
    let nlayers = board.spec.nlayers;
    let before = board.vias.len();
    let mut by_key: HashMap<(i64, i64, u32), Merge> = HashMap::new();
    for v in board.vias.drain(..) {
        let key = (v.pos.x.0, v.pos.y.0, v.net.0);
        by_key
            .entry(key)
            .and_modify(|(lo, hi, filled, _)| {
                *lo = (*lo).min(v.from.0).min(v.to.0);
                *hi = (*hi).max(v.from.0).max(v.to.0);
                *filled = *filled || v.filled; // a merged stack stays plated-over if any leg was
            })
            .or_insert((
                v.from.0.min(v.to.0),
                v.from.0.max(v.to.0),
                v.filled,
                v,
            ));
    }
    let mut out: Vec<crate::board::Via> = by_key
        .into_values()
        .map(|(lo, hi, filled, mut v)| {
            // The merged span may now be a deeper pathway (e.g. two micro legs -> a blind-via), so
            // re-derive the construction class and the matching drill/diameter under the board policy.
            let (from, to, kind, drill, diameter) = rules.resolve_via(lo, hi, nlayers);
            v.from = from;
            v.to = to;
            v.filled = filled;
            v.kind = kind;
            v.drill = drill;
            v.diameter = diameter;
            v
        })
        .collect();
    out.sort_by_key(|v| (v.pos.x.0, v.pos.y.0, v.net.0));
    board.vias = out;
    before.saturating_sub(board.vias.len())
}

///
/// A teardrop is a tapered copper transition from the feature's pad down to the track width; it
/// keeps the track connected to the pad even if the drill wanders (registration error eats the
/// annular ring on one side), the dominant bare-board open-fault mechanism (IPC-2221 annular-ring
/// concern). Each teardrop is a solid same-net [`Zone`]; the filler trims it to clearance, so it
/// reinforces where there is room and is trimmed where there is not. Returns the count added.
///
/// Teardrop size comes from each feature's own geometry (radius and the track width). SMD pads are
/// skipped (no drill ⇒ no registration risk). One teardrop is emitted per track approaching a
/// feature's centre.
pub fn teardrops(board: &mut Board) -> usize {
    let tol = board.spec.pitch.0 as f64;
    let mut added = 0;

    // Collect drilled features as (centre, radius, net). Vias always; pads spanning >1 layer are
    // thru-hole (the io layer rule), so they are drilled too.
    let mut feats: Vec<(Point, f64, NetId, LayerId)> = Vec::new();
    for v in &board.vias {
        feats.push((v.pos, v.diameter.0 as f64 / 2.0, v.net, v.from));
    }
    for p in &board.pads {
        if p.layers.len() > 1 {
            if let Some(n) = p.net {
                // Pad radius unknown here (Pad has no size); use a via-class radius as a safe floor.
                feats.push((p.pos, board.spec.pitch.0 as f64 * 0.45, n, p.layers[0]));
            }
        }
    }

    // Index tracks by layer for the approach search.
    for (centre, radius, net, _flayer) in feats {
        for t in &board.tracks {
            if t.net != net {
                continue;
            }
            // Which endpoint (if any) lands on the feature?
            let at_start = t.start.euclid(centre) <= tol;
            let at_end = t.end.euclid(centre) <= tol;
            if !at_start && !at_end {
                continue;
            }
            let (near, far) = if at_start {
                (t.start, t.end)
            } else {
                (t.end, t.start)
            };
            let seg_len = near.euclid(far);
            if seg_len < tol {
                continue;
            }
            // Unit direction along the track (away from the feature) and its perpendicular.
            let (ux, uy) = (
                (far.x.0 - near.x.0) as f64 / seg_len,
                (far.y.0 - near.y.0) as f64 / seg_len,
            );
            let (nx, ny) = (-uy, ux);
            let l = radius.mul_add(2.0, 0.0).min(seg_len); // teardrop length ≈ 2·R, clipped to track
            let hw = t.width.0 as f64 / 2.0;
            let cx = centre.x.0 as f64;
            let cy = centre.y.0 as f64;
            let pt = |dx: f64, dy: f64| Point::new(Nm((cx + dx) as i64), Nm((cy + dy) as i64));
            // Trapezoid: wide (±R) at the pad, narrowing to ±half-track-width at distance L.
            let poly = vec![
                pt(nx * radius, ny * radius),
                pt(ux * l + nx * hw, uy * l + ny * hw),
                pt(ux * l - nx * hw, uy * l - ny * hw),
                pt(-nx * radius, -ny * radius),
            ];
            board.zones.push(Zone {
                net,
                layer: t.layer,
                polygon: poly,
                fill: ZoneFill::Solid,
            });
            added += 1;
        }
    }
    added
}

/// Distribute a plane-backed net through its copper pour instead of routed tracks.
///
/// High-current power and ground rails (VPP, GND) belong on **planes**, not thin routed traces:
/// the pour carries the current (ampacity via copper area), references the signals for EMI return,
/// and — critically — a routed power *track* sitting next to a foreign signal track is a clearance
/// short the grid router cannot see. This pass deletes every routed track **and** via of `net`
/// (the pour replaces them), then drops a stitching via from each of the net's pads that does not
/// already land on a poured layer down to the nearest plane in `plane_layers`, tying the pad into
/// the pour. Returns `(tracks_removed, stitch_vias_added)`.
///
/// Precondition: `net` is already poured on every layer in `plane_layers` (e.g. via
/// [`ground_pour`]). Inner-layer pours are full-board, so a stitching via to an inner plane always
/// lands in copper; an outer pour covers the net's own feature hull. A pad already on a poured
/// layer (a thru-hole barrel, or an SMD pad on an outer poured layer) needs no via — the filler
/// connects it directly.
pub fn plane_distribute_net(
    board: &mut Board,
    net: NetId,
    plane_layers: &[LayerId],
    rules: &DesignRules,
) -> (usize, usize) {
    let before_tracks = board.tracks.len();
    board.tracks.retain(|t| t.net != net);
    let removed = before_tracks - board.tracks.len();
    // The net's routed vias are superseded by the pour + stitching vias re-derived below.
    board.vias.retain(|v| v.net != net);

    let nlayers = board.spec.nlayers;
    let pads: Vec<(Point, Vec<LayerId>)> = board
        .pads
        .iter()
        .filter(|p| p.net == Some(net))
        .map(|p| (p.pos, p.layers.clone()))
        .collect();
    let mut added = 0;
    for (pos, layers) in pads {
        // Already tied to the pour: the pad sits on a poured layer (thru-hole barrel spans them all).
        if layers.iter().any(|l| plane_layers.contains(l)) {
            continue;
        }
        let from = layers.iter().map(|l| l.0).min().unwrap_or(0);
        let Some(target) = plane_layers
            .iter()
            .map(|l| l.0)
            .min_by_key(|&pl| (pl as i32 - from as i32).unsigned_abs())
        else {
            continue; // no plane to stitch to
        };
        let (lo, hi) = (from.min(target), from.max(target));
        let (vf, vt, kind, drill, diameter) = rules.resolve_via(lo, hi, nlayers);
        board.vias.push(crate::board::Via {
            pos,
            drill,
            diameter,
            net,
            from: vf,
            to: vt,
            kind,
            filled: false,
        });
        added += 1;
    }
    (removed, added)
}
