//! Net query utilities — shared helpers for routing, diff-pair, and high-speed detectors.
//!
//! All items are `pub(crate)`. Only detectors and the `audit()` orchestrator call these.

use std::collections::{BTreeMap, HashMap};

use crate::board::{Board, LayerId, NetClassKind, NetId, Track};
use crate::geom::{Nm, Point, point_in_polygon};

/// Midpoint of a track segment.
pub(crate) fn track_midpoint(track: &Track) -> Point {
    Point::new(
        Nm((track.start.x.0 + track.end.x.0) / 2),
        Nm((track.start.y.0 + track.end.y.0) / 2),
    )
}

/// Cumulative routed length per layer for a set of tracks.
pub(crate) fn diff_pair_layer_segment_lengths(tracks: &[&Track]) -> BTreeMap<LayerId, (f64, Option<Point>)> {
    let mut lengths = BTreeMap::new();
    for track in tracks {
        let entry = lengths
            .entry(track.layer)
            .or_insert_with(|| (0.0, Some(track_midpoint(track))));
        entry.0 += track.start.euclid(track.end);
    }
    lengths
}

/// Iterator over series capacitors connected to a net (used to locate diff-pair coupling caps).
pub(crate) fn diff_pair_coupling_caps(
    comps: &[crate::place::Component],
    net: NetId,
) -> impl Iterator<Item = &crate::place::Component> {
    comps
        .iter()
        .filter(move |c| c.refdes.starts_with('C') && c.nets.contains(&Some(net)))
}

/// Distance from each pad on `net` to the nearest track endpoint on the same net.
pub(crate) fn diff_pair_pad_entry_distances(board: &Board, net: NetId) -> Vec<(f64, Point)> {
    let endpoints: Vec<Point> = board
        .tracks
        .iter()
        .filter(|track| track.net == net)
        .flat_map(|track| [track.start, track.end])
        .collect();
    if endpoints.is_empty() {
        return Vec::new();
    }

    let mut distances: Vec<(f64, Point)> = board
        .pads_of(net)
        .map(|pad| {
            let nearest = endpoints
                .iter()
                .map(|&endpoint| pad.pos.euclid(endpoint))
                .fold(f64::INFINITY, f64::min);
            (nearest, pad.pos)
        })
        .collect();
    distances.sort_by(|a, b| a.0.total_cmp(&b.0));
    distances
}

/// Returns the bus-group prefix for a net name like `"DATA_BUS7"` → `Some("DATA_BUS")`,
/// or `None` if the name is not a numbered bus signal.
pub(crate) fn parallel_bus_group_name(name: &str) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    let trimmed = upper.trim_end_matches(|c: char| c.is_ascii_digit());
    if trimmed.len() == upper.len() || !trimmed.contains("BUS") {
        return None;
    }
    let group = trimmed.trim_end_matches(['_', '-', '[']);
    (!group.is_empty()).then(|| group.to_string())
}

/// Whether `net` is classified as high-speed (HV class, TRIG, OUT, or TX prefix).
pub(crate) fn is_high_speed_net(board: &Board, net: NetId) -> bool {
    let name = &board.nets[net.0 as usize].name;
    board.class_of(net) == NetClassKind::Hv
        || name.starts_with("TRIG")
        || name.starts_with("OUT")
        || name.starts_with("TX")
}

/// Whether `net` has a clock-like name (CLK, _CLK, CLOCK).
pub(crate) fn is_clock_like_net(board: &Board, net: NetId) -> bool {
    let upper = board.nets[net.0 as usize].name.to_ascii_uppercase();
    upper.starts_with("CLK") || upper.contains("_CLK") || upper.contains("CLOCK")
}

/// Strips the polarity suffix from a diff-pair net name, returning the common prefix.
pub(crate) fn diff_pair_prefix(name: &str) -> Option<&str> {
    name.strip_suffix("_P")
        .or_else(|| name.strip_suffix("_N"))
        .or_else(|| name.strip_suffix('+'))
        .or_else(|| name.strip_suffix('-'))
        .or_else(|| name.strip_suffix("_pos"))
        .or_else(|| name.strip_suffix("_neg"))
}

/// Returns `true` if `a` and `b` are the P and N members of the same diff pair.
pub(crate) fn are_diff_pair_mates(board: &Board, a: NetId, b: NetId) -> bool {
    let an = &board.nets[a.0 as usize].name;
    let bn = &board.nets[b.0 as usize].name;
    let Some(ap) = diff_pair_prefix(an) else {
        return false;
    };
    let Some(bp) = diff_pair_prefix(bn) else {
        return false;
    };
    ap == bp && an != bn
}

/// Collects all (P-net, N-net) diff-pair pairs present on the board.
pub(crate) fn diff_pair_members(board: &Board) -> Vec<(NetId, NetId)> {
    let mut p_nets = Vec::new();
    let mut n_nets = Vec::new();
    for net in &board.nets {
        let name = &net.name;
        if name.ends_with("_P") || name.ends_with('+') || name.ends_with("_pos") {
            p_nets.push(net);
        } else if name.ends_with("_N") || name.ends_with('-') || name.ends_with("_neg") {
            n_nets.push(net);
        }
    }

    let mut pairs = Vec::new();
    for p in &p_nets {
        let Some(prefix) = diff_pair_prefix(&p.name) else {
            continue;
        };
        for n in &n_nets {
            if diff_pair_prefix(&n.name) == Some(prefix) {
                pairs.push((p.id, n.id));
                break;
            }
        }
    }
    pairs
}

/// Returns the interface group name for a diff-pair prefix such as `"PCIE_TX"` → `Some("PCIE_TX")`,
/// stripping trailing digits that identify the lane index.
pub(crate) fn diff_pair_interface_group(prefix: &str) -> Option<String> {
    let upper = prefix.to_ascii_uppercase();
    let trimmed = upper.trim_end_matches(|c: char| c.is_ascii_digit());
    if trimmed.len() == upper.len() {
        return None;
    }
    let group = trimmed.trim_end_matches(['_', '-', '[']);
    (!group.is_empty()).then(|| group.to_string())
}

/// Maps each member of each diff pair to its pair index for fast lookups.
pub(crate) fn diff_pair_member_to_pair(pairs: &[(NetId, NetId)]) -> HashMap<NetId, usize> {
    let mut out = HashMap::new();
    for (idx, &(p, n)) in pairs.iter().enumerate() {
        out.insert(p, idx);
        out.insert(n, idx);
    }
    out
}

/// Reference-plane zones (Ground or Power class) used by high-speed and diff-pair detectors.
pub(crate) fn reference_zones(board: &Board) -> Vec<&crate::board::Zone> {
    board
        .zones
        .iter()
        .filter(|z| {
            let class = board.class_of(z.net);
            matches!(class, NetClassKind::Ground | NetClassKind::Power)
        })
        .collect()
}

/// Returns the power-reference zone adjacent to `track` if no ground reference exists adjacent,
/// or `None` if the track has a ground reference or no reference at all.
pub(crate) fn power_reference_zone_for_track<'a>(
    board: &Board,
    zones: &'a [&'a crate::board::Zone],
    track: &Track,
) -> Option<&'a crate::board::Zone> {
    let mid = track_midpoint(track);
    let samples = [track.start, mid, track.end];
    let has_ground_reference = zones.iter().any(|zone| {
        board.class_of(zone.net) == NetClassKind::Ground
            && zone.layer.0.abs_diff(track.layer.0) == 1
            && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
    });
    if has_ground_reference {
        return None;
    }
    zones.iter().copied().find(|zone| {
        board.class_of(zone.net) == NetClassKind::Power
            && zone.layer.0.abs_diff(track.layer.0) == 1
            && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon))
    })
}

/// Indices (into the `reference_zones(board)` slice) of ground reference zones adjacent to `track`.
pub(crate) fn adjacent_ground_reference_zone_indices(board: &Board, track: &Track) -> Vec<usize> {
    let zones = reference_zones(board);
    let mid = track_midpoint(track);
    let samples = [track.start, mid, track.end];
    zones
        .iter()
        .enumerate()
        .filter_map(|(idx, zone)| {
            (board.class_of(zone.net) == NetClassKind::Ground
                && zone.layer.0.abs_diff(track.layer.0) == 1
                && samples.iter().all(|&p| point_in_polygon(p, &zone.polygon)))
            .then_some(idx)
        })
        .collect()
}
