//! The 90° → 135° corner-mitering DFM pass: insert a short 45° chamfer at each right-angle
//! same-net track junction (NASA HS guide §7.3.1.3 prefers 135° bends), guarded against landing a
//! chamfer endpoint inside a foreign pad halo.

use crate::board::{Board, NetId, Track};
use crate::geom::{dist_point_seg, Nm, Point};
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::rules::DesignRules;

/// Replace right-angle (90°) junctions between axial track segments with 135°–45°–135°
/// mitered bends by inserting a short 45° diagonal chamfer at each corner.
///
/// For every apex P shared by a horizontal and a vertical same-net same-layer track pair, this
/// pass emits:
/// - A shortened horizontal stub from the H arm's far end to `P1 = P ± chamfer` along the H arm.
/// - A 45° diagonal from `P1` to `P2 = P ± chamfer` along the V arm.
/// - A shortened vertical stub from `P2` to the V arm's far end.
///
/// Both resulting junctions at `P1` and `P2` subtend 135°, which is the preferred DFM bend
/// geometry (IPC-2221 §10.4.3: only angles **< 90°** are acid-trap concerns). Junctions at
/// T-intersections (≥ 3 tracks meeting) or where either arm is shorter than `chamfer` are left
/// unchanged. Each apex is processed at most once; a track shared by two apices is shortened
/// only at the first apex encountered.
///
/// Returns the number of corners mitered. Call [`crate::dfm::merge_collinear`] afterwards to fold any
/// collinear stub pairs the chamfer may create (e.g. when P1 abuts another collinear H segment).
/// Replace 90° H+V track junctions with mitered 135°–45°–135° bends.
///
/// `comps` + `lib` + `rules` supply the pad-proximity guard: a miter whose 45° diagonal would
/// come within `min_clearance + half_track_width` of any foreign-net or unconnected pad edge is
/// silently skipped, preserving the original right-angle corner rather than emitting a short.
pub fn miter_right_angle_corners(
    board: &mut Board,
    chamfer: Nm,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> usize {
    if chamfer.0 <= 0 {
        return 0;
    }
    let cd = chamfer.0; // chamfer distance in nm

    // Pre-compute all-board pad obstacles for the pad-proximity guard on miter endpoints.
    // Each entry: (pad centre, half-diagonal in nm, net-or-None).
    // Same conservative geometry as pad_entry_stubs (half-diagonal = hypot(w, h) / 2).
    let all_pads: Vec<(Point, f64, Option<NetId>)> = comps
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

    // Map endpoint (x_nm, y_nm, layer, net) → Vec<(track_idx, is_start)>.
    type EpKey = (i64, i64, u16, u32);
    let mut ep_map: std::collections::HashMap<EpKey, Vec<(usize, bool)>> =
        std::collections::HashMap::new();
    for (i, t) in board.tracks.iter().enumerate() {
        let ks = (t.start.x.0, t.start.y.0, t.layer.0, t.net.0);
        let ke = (t.end.x.0, t.end.y.0, t.layer.0, t.net.0);
        ep_map.entry(ks).or_default().push((i, true));
        ep_map.entry(ke).or_default().push((i, false));
    }

    // Collect miters: (h_idx, h_is_start_apex, v_idx, v_is_start_apex, P1, P2, track_width).
    let mut miters: Vec<(usize, bool, usize, bool, Point, Point, Nm)> = Vec::new();
    let mut processed: std::collections::HashSet<EpKey> = std::collections::HashSet::new();

    for (&key, endpoints) in &ep_map {
        if endpoints.len() != 2 || processed.contains(&key) {
            continue;
        }
        let (ai, a_is_start) = endpoints[0];
        let (bi, b_is_start) = endpoints[1];
        let ta = &board.tracks[ai];
        let tb = &board.tracks[bi];

        let apex = if a_is_start { ta.start } else { ta.end };
        let a_far = if a_is_start { ta.end } else { ta.start };
        let b_far = if b_is_start { tb.end } else { tb.start };

        let adx = (a_far.x.0 - apex.x.0).abs();
        let ady = (a_far.y.0 - apex.y.0).abs();
        let bdx = (b_far.x.0 - apex.x.0).abs();
        let bdy = (b_far.y.0 - apex.y.0).abs();

        // One arm must be pure horizontal (dy=0, dx>0) and the other pure vertical (dx=0, dy>0).
        let (h_idx, h_is_start, h_far, v_idx, v_is_start, v_far) =
            if adx != 0 && ady == 0 && bdx == 0 && bdy != 0 {
                (ai, a_is_start, a_far, bi, b_is_start, b_far)
            } else if adx == 0 && ady != 0 && bdx != 0 && bdy == 0 {
                (bi, b_is_start, b_far, ai, a_is_start, a_far)
            } else {
                continue; // diagonal or same-axis: not a right-angle corner
            };

        // Adaptive chamfer: a symmetric 45° miter needs each arm ≥ the chamfer. The preferred size is
        // `cd` (the signal-track width — the IPC chamfer reference), but on a fine-pitch escape grid
        // many arms are only one or two cells long, so a fixed `cd` would skip them and leave a 90°
        // reflection point (NASA HS guide §7.3.1.3: use 135° bends, not 90°). Shrink the chamfer to at
        // most 40 % of the shorter arm so tight corners are still mitered; below a small floor a
        // sub-fab-resolution diagonal is not worth emitting and the corner is left as-is.
        let h_len = (h_far.x.0 - apex.x.0).abs();
        let v_len = (v_far.y.0 - apex.y.0).abs();
        const MIN_CHAMFER_NM: i64 = 50_000; // 0.05 mm — below this a 45° chamfer is plot noise
        let eff_cd = cd.min(h_len.min(v_len) * 2 / 5);
        if eff_cd < MIN_CHAMFER_NM {
            continue;
        }

        // P1 is on the H arm, chamfer distance from apex toward h_far.
        let h_sign: i64 = if h_far.x.0 > apex.x.0 { 1 } else { -1 };
        let v_sign: i64 = if v_far.y.0 > apex.y.0 { 1 } else { -1 };
        let p1 = Point::new(Nm(apex.x.0 + h_sign * eff_cd), apex.y);
        let p2 = Point::new(apex.x, Nm(apex.y.0 + v_sign * eff_cd));

        // Pad-proximity guard: skip this miter when the 45° diagonal P1→P2 would come
        // within (min_clearance + half_track_width) of any foreign-net or unconnected pad
        // copper edge. Prevents miter endpoints landing inside pad halos — root cause of
        // a class of DFM-pass clearance violations on fine-pitch dense routing.
        if !all_pads.is_empty() {
            let track_net = board.tracks[h_idx].net;
            let half_w = (board.tracks[h_idx].width.0 / 2) as f64;
            let guard = rules.min_clearance.0 as f64 + half_w;
            let blocked = all_pads.iter().any(|&(opos, half_diag, onet)| {
                onet != Some(track_net) && dist_point_seg(opos, p1, p2) < half_diag + guard
            });
            if blocked {
                continue;
            }
        }

        processed.insert(key);
        let width = board.tracks[h_idx].width;
        miters.push((h_idx, h_is_start, v_idx, v_is_start, p1, p2, width));
    }

    if miters.is_empty() {
        return 0;
    }

    let count = miters.len();
    #[derive(Debug, Default)]
    struct TrackPatch {
        start: Option<Point>,
        end: Option<Point>,
    }
    let mut mods: std::collections::HashMap<usize, TrackPatch> = std::collections::HashMap::new();
    let mut additions: Vec<Track> = Vec::new();

    for (h_idx, h_is_start, v_idx, v_is_start, p1, p2, width) in miters {
        let h_track = board.tracks[h_idx];

        // Shorten each track at the apex end. A segment can legitimately have right-angle corners
        // at both ends; patch start/end independently so the second miter's diagonal lands on a real
        // segment endpoint instead of in the middle of an unmodified track body.
        let h_patch = mods.entry(h_idx).or_default();
        if h_is_start {
            h_patch.start = Some(p1);
        } else {
            h_patch.end = Some(p1);
        }

        let v_patch = mods.entry(v_idx).or_default();
        if v_is_start {
            v_patch.start = Some(p2);
        } else {
            v_patch.end = Some(p2);
        }

        additions.push(Track {
            start: p1,
            end: p2,
            width,
            layer: h_track.layer,
            net: h_track.net,
        });
    }

    // Rebuild: apply modifications, drop zero-length tracks, append diagonals.
    board.tracks = board
        .tracks
        .drain(..)
        .enumerate()
        .filter_map(|(i, t)| {
            let t2 = if let Some(patch) = mods.remove(&i) {
                Track {
                    start: patch.start.unwrap_or(t.start),
                    end: patch.end.unwrap_or(t.end),
                    ..t
                }
            } else {
                t
            };
            if t2.start == t2.end {
                None
            } else {
                Some(t2)
            }
        })
        .collect();
    board.tracks.extend(additions);
    count
}
