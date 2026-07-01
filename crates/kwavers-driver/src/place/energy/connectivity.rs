//! HPWL, net-centres, flight-lines, signal-flow, and channel-blockage energy terms.
//!
//! Extracted from `compute::energy()` — lines 283–424 of the original `compute.rs`.
//! All arithmetic is bit-for-bit identical to the original; no logic was changed.

use std::collections::BTreeMap;

use super::config::{EnergyTerms, PlaceConfig};
use super::geom::{
    connector_ingress_unit, has_non_power_pad_on_net, non_power_signal_net_count,
    segment_intersects_rect,
};
use crate::board::NetId;
use crate::geom::{segments_cross, Nm, Point};
use crate::place::component::Component;
use crate::place::footprint::{FootprintDef, Role};

/// A two-component local net flight line (used only inside [`accumulate_connectivity`]).
struct FlightLine {
    members: [usize; 2],
    a: Point,
    b: Point,
}

/// Accumulate HPWL, net-centre flight-line, signal-flow, crossing, and channel-blockage terms.
///
/// # Arguments
/// * `t` — energy accumulator; `hpwl`, `regional`, `flow_crossing`, and `channel_blockage` are
///   updated.
/// * `comps` — full component list.
/// * `lib` — footprint library.
/// * `cfg` — placement configuration (unused directly but kept for API symmetry).
/// * `half_clear` — half the courtyard clearance (for channel keepout inflation).
/// * `w` / `h` — board width/height in mm.
/// * `board_diag` — board diagonal in mm (pre-computed by caller to avoid recomputation).
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_connectivity(
    t: &mut EnergyTerms,
    comps: &[Component],
    lib: &[FootprintDef],
    _cfg: &PlaceConfig,
    half_clear: Nm,
    w: f64,
    h: f64,
    board_diag: f64,
) {
    // HPWL: per-net bounding-box half-perimeter. BTreeMap (not HashMap) so the f64 summation
    // order is deterministic — otherwise randomised iteration perturbs the total and flips SA
    // accept/reject decisions, making placement non-reproducible.
    let mut bbox: BTreeMap<NetId, (f64, f64, f64, f64)> = BTreeMap::new();
    for c in comps {
        for (pos, _layers, net) in c.placed_pads(lib) {
            if let Some(n) = net {
                let x = pos.x.to_mm();
                let y = pos.y.to_mm();
                let e = bbox.entry(n).or_insert((x, y, x, y));
                e.0 = e.0.min(x);
                e.1 = e.1.min(y);
                e.2 = e.2.max(x);
                e.3 = e.3.max(y);
            }
        }
    }
    for (_n, (x1, y1, x2, y2)) in bbox {
        t.hpwl += (x2 - x1) + (y2 - y1);
    }

    let mut net_centers: BTreeMap<NetId, Vec<(usize, Point)>> = BTreeMap::new();
    // Hoisted outside the loop so each component iteration reuses the allocation.
    // Typical capacity: 2–8 entries (one per distinct net on the component's pads).
    let mut sum_by_net: Vec<(NetId, (i64, i64, i64))> = Vec::with_capacity(8);
    for (idx, c) in comps.iter().enumerate() {
        // Each component typically has 2–8 pads across 1–4 nets; a Vec linear scan is
        // faster than a BTreeMap allocation for this small N (no heap per-node overhead).
        sum_by_net.clear();
        for (pos, _layers, net) in c.placed_pads(lib) {
            if let Some(n) = net {
                if let Some(e) = sum_by_net.iter_mut().find(|(id, _)| *id == n) {
                    e.1 .0 += pos.x.0;
                    e.1 .1 += pos.y.0;
                    e.1 .2 += 1;
                } else {
                    sum_by_net.push((n, (pos.x.0, pos.y.0, 1)));
                }
            }
        }
        // Sort by NetId before inserting so net_centers iteration order matches the outer
        // BTreeMap's determinism requirement (same insertion order → same HPWL sum).
        sum_by_net.sort_unstable_by_key(|(id, _)| *id);
        for (net, (sx, sy, count)) in sum_by_net.drain(..) {
            net_centers
                .entry(net)
                .or_default()
                .push((idx, Point::new(Nm(sx / count), Nm(sy / count))));
        }
    }

    let mut flight_lines: Vec<FlightLine> = Vec::new();
    let mut local_flow_vectors: Vec<Vec<(f64, f64)>> = vec![Vec::new(); comps.len()];
    for (net, centers) in &net_centers {
        if centers.len() == 2 {
            let a = centers[0].0;
            let b = centers[1].0;
            let a_fp = &lib[comps[a].fp];
            let b_fp = &lib[comps[b].fp];
            if matches!(a_fp.role, Role::ActiveIc)
                && matches!(b_fp.role, Role::ActiveIc)
                && non_power_signal_net_count(&comps[a], a_fp) == 1
                && non_power_signal_net_count(&comps[b], b_fp) == 1
                && has_non_power_pad_on_net(&comps[a], a_fp, *net)
                && has_non_power_pad_on_net(&comps[b], b_fp, *net)
            {
                t.regional += centers[0].1.euclid(centers[1].1) * 1.0e-6;
            }
            let ax = comps[a].placement.pos.x.to_mm();
            let ay = comps[a].placement.pos.y.to_mm();
            let bx = comps[b].placement.pos.x.to_mm();
            let by = comps[b].placement.pos.y.to_mm();
            local_flow_vectors[a].push((bx - ax, by - ay));
            local_flow_vectors[b].push((ax - bx, ay - by));
            flight_lines.push(FlightLine {
                members: [centers[0].0, centers[1].0],
                a: centers[0].1,
                b: centers[1].1,
            });
            let a_connector = matches!(lib[comps[a].fp].role, Role::Connector);
            let b_connector = matches!(lib[comps[b].fp].role, Role::Connector);
            if a_connector != b_connector {
                let (connector, other) = if a_connector { (a, b) } else { (b, a) };
                let cx = comps[connector].placement.pos.x.to_mm();
                let cy = comps[connector].placement.pos.y.to_mm();
                let ox = comps[other].placement.pos.x.to_mm();
                let oy = comps[other].placement.pos.y.to_mm();
                let dx = ox - cx;
                let dy = oy - cy;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.0 {
                    let (ux, uy) = connector_ingress_unit(comps[connector].placement.pos, w, h);
                    let cosine = ((dx * ux + dy * uy) / len).clamp(-1.0, 1.0);
                    t.regional += 1.0 - cosine;
                }
            }
        }
    }
    // Regional signal flow: at a component joining two local point-to-point nets, both neighbors on
    // the same side form a fold-back and orthogonal neighbors form a dogleg instead of the smooth
    // unidirectional flow expected from a floorplanned high-speed subsection. The normalized dot
    // product makes a straight fold-back cost 1.0, and the normalized cross product makes a right
    // angle cost 1.0 while collinear through-flow costs 0.
    for vectors in &local_flow_vectors {
        for i in 0..vectors.len() {
            let (ax, ay) = vectors[i];
            let a_len = (ax * ax + ay * ay).sqrt();
            for &(bx, by) in vectors.iter().skip(i + 1) {
                let b_len = (bx * bx + by * by).sqrt();
                if a_len == 0.0 || b_len == 0.0 {
                    continue;
                }
                let direction_cosine = (ax * bx + ay * by) / (a_len * b_len);
                if direction_cosine > 0.0 {
                    t.regional += direction_cosine;
                }
                t.regional += ((ax * by - ay * bx) / (a_len * b_len)).abs();
            }
        }
    }
    for i in 0..flight_lines.len() {
        for other in flight_lines.iter().skip(i + 1) {
            let line = &flight_lines[i];
            if segments_cross(line.a, line.b, other.a, other.b) {
                t.flow_crossing += 1.0;
            }
        }
        let line = &flight_lines[i];
        for (idx, c) in comps.iter().enumerate() {
            if line.members.contains(&idx) {
                continue;
            }
            let channel_keepout = c.courtyard(lib).inflate(half_clear);
            if segment_intersects_rect(line.a, line.b, channel_keepout) {
                t.channel_blockage += 1.0;
            }
        }
    }

    // board_diag is part of the public signature for API symmetry with the other accumulators
    // (the caller pre-computes it to avoid redundant sqrt) but is not used by the present
    // implementation.
    let _ = board_diag;
}
