//! Split-domain reference crossing, mixed-domain shared reference, and
//! virtual split-line crossing violation detectors.

use crate::board::{split_domain_from_name, Board, NetClassKind, SplitDomain};
use crate::geom::{dist_seg_seg, point_in_polygon, Nm, Point};
use crate::rules::DesignRules;
use crate::audit::net_util::{
    adjacent_ground_reference_zone_indices, reference_zones, track_midpoint,
};

pub(crate) fn detect_split_domain_reference_violations(board: &Board) -> (usize, Vec<Point>) {
    let zones = reference_zones(board);
    let mut count = 0;
    let mut pts = Vec::new();

    for track in &board.tracks {
        if matches!(
            board.class_of(track.net),
            NetClassKind::Ground | NetClassKind::Power
        ) {
            continue;
        }
        let signal_name = &board.nets[track.net.0 as usize].name;
        let Some(signal_domain) = split_domain_from_name(signal_name) else {
            continue;
        };
        let mid = track_midpoint(track);
        for zone in zones.iter().filter(|zone| zone.layer == track.layer) {
            let reference_name = &board.nets[zone.net.0 as usize].name;
            let Some(reference_domain) = split_domain_from_name(reference_name) else {
                continue;
            };
            if signal_domain != reference_domain && point_in_polygon(mid, &zone.polygon) {
                count += 1;
                pts.push(mid);
                break;
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_mixed_domain_shared_reference_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let mut domain_tracks = Vec::new();
    for track in &board.tracks {
        if board.class_of(track.net) != NetClassKind::Signal {
            continue;
        }
        let signal_name = &board.nets[track.net.0 as usize].name;
        let Some(domain) = split_domain_from_name(signal_name) else {
            continue;
        };
        let references = adjacent_ground_reference_zone_indices(board, track);
        if !references.is_empty() {
            domain_tracks.push((track, domain, references));
        }
    }

    let mut count = 0;
    let mut pts = Vec::new();
    let max_gap = rules.diff_pair_clock_keepout.0 as f64;
    for i in 0..domain_tracks.len() {
        let (a, a_domain, a_refs) = &domain_tracks[i];
        for (b, b_domain, b_refs) in domain_tracks.iter().skip(i + 1) {
            if a_domain == b_domain || !a_refs.iter().any(|r| b_refs.contains(r)) {
                continue;
            }
            let copper_gap =
                dist_seg_seg(a.start, a.end, b.start, b.end) - (a.width.0 + b.width.0) as f64 / 2.0;
            if copper_gap <= max_gap {
                let am = track_midpoint(a);
                let bm = track_midpoint(b);
                count += 1;
                pts.push(Point::new(
                    Nm((am.x.0 + bm.x.0) / 2),
                    Nm((am.y.0 + bm.y.0) / 2),
                ));
            }
        }
    }

    (count, pts)
}

pub(crate) fn detect_virtual_split_crossing_violations(board: &Board) -> (usize, Vec<Point>) {
    let mut analog_sum = (0_i64, 0_i64, 0_i64);
    let mut digital_sum = (0_i64, 0_i64, 0_i64);
    for pad in &board.pads {
        let Some(net) = pad.net else { continue };
        match split_domain_from_name(&board.nets[net.0 as usize].name) {
            Some(SplitDomain::Analog) => {
                analog_sum.0 += pad.pos.x.0;
                analog_sum.1 += pad.pos.y.0;
                analog_sum.2 += 1;
            }
            Some(SplitDomain::Digital) => {
                digital_sum.0 += pad.pos.x.0;
                digital_sum.1 += pad.pos.y.0;
                digital_sum.2 += 1;
            }
            None => {}
        }
    }
    if analog_sum.2 == 0 || digital_sum.2 == 0 {
        return (0, Vec::new());
    }

    let analog = Point::new(
        Nm(analog_sum.0 / analog_sum.2),
        Nm(analog_sum.1 / analog_sum.2),
    );
    let digital = Point::new(
        Nm(digital_sum.0 / digital_sum.2),
        Nm(digital_sum.1 / digital_sum.2),
    );
    let use_x_axis = (analog.x.0 - digital.x.0).abs() >= (analog.y.0 - digital.y.0).abs();
    let split = if use_x_axis {
        (analog.x.0 + digital.x.0) / 2
    } else {
        (analog.y.0 + digital.y.0) / 2
    };
    let analog_less = if use_x_axis {
        analog.x.0 < split
    } else {
        analog.y.0 < split
    };

    let mut count = 0;
    let mut pts = Vec::new();
    for track in &board.tracks {
        if board.class_of(track.net) != NetClassKind::Signal {
            continue;
        }
        let Some(domain) = split_domain_from_name(&board.nets[track.net.0 as usize].name) else {
            continue;
        };
        let expected_less = match domain {
            SplitDomain::Analog => analog_less,
            SplitDomain::Digital => !analog_less,
        };
        let a = if use_x_axis {
            track.start.x.0
        } else {
            track.start.y.0
        };
        let b = if use_x_axis {
            track.end.x.0
        } else {
            track.end.y.0
        };
        let start_ok = if expected_less { a < split } else { a > split };
        let end_ok = if expected_less { b < split } else { b > split };
        if !(start_ok && end_ok) {
            count += 1;
            pts.push(track_midpoint(track));
        }
    }

    (count, pts)
}
