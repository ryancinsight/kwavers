//! Geometry helpers shared between the placement-energy penalty terms.

use std::collections::BTreeSet;

use crate::board::NetId;
use crate::geom::{segments_cross, Nm, Point};
use crate::place::component::{Component, Rect};
use crate::place::footprint::FootprintDef;
use crate::place::rotation::Rot;

pub(crate) fn rotation_axis(rot: Rot) -> u8 {
    match rot {
        Rot::R0 | Rot::R180 => 0,
        Rot::R90 | Rot::R270 => 1,
    }
}

pub(crate) fn rect_contains_point(rect: Rect, p: Point) -> bool {
    (rect.min.x.0..=rect.max.x.0).contains(&p.x.0)
        && (rect.min.y.0..=rect.max.y.0).contains(&p.y.0)
}

pub(crate) fn segment_intersects_rect(a: Point, b: Point, rect: Rect) -> bool {
    if rect_contains_point(rect, a) || rect_contains_point(rect, b) {
        return true;
    }
    let p0 = rect.min;
    let p1 = Point::new(rect.max.x, rect.min.y);
    let p2 = rect.max;
    let p3 = Point::new(rect.min.x, rect.max.y);
    segments_cross(a, b, p0, p1)
        || segments_cross(a, b, p1, p2)
        || segments_cross(a, b, p2, p3)
        || segments_cross(a, b, p3, p0)
}

pub(crate) fn rect_gap_mm(a: Rect, b: Rect) -> f64 {
    let ax_min = a.min.x.to_mm();
    let ax_max = a.max.x.to_mm();
    let ay_min = a.min.y.to_mm();
    let ay_max = a.max.y.to_mm();
    let bx_min = b.min.x.to_mm();
    let bx_max = b.max.x.to_mm();
    let by_min = b.min.y.to_mm();
    let by_max = b.max.y.to_mm();
    let dx = if ax_max < bx_min {
        bx_min - ax_max
    } else if bx_max < ax_min {
        ax_min - bx_max
    } else {
        0.0
    };
    let dy = if ay_max < by_min {
        by_min - ay_max
    } else if by_max < ay_min {
        ay_min - by_max
    } else {
        0.0
    };
    (dx * dx + dy * dy).sqrt()
}

pub(crate) fn carries_connected_signal(c: &Component, fp: &FootprintDef) -> bool {
    c.nets
        .iter()
        .zip(fp.pads.iter())
        .any(|(net, pad)| net.is_some() && !pad.power_pin)
}

pub(crate) fn has_non_power_pad_on_net(c: &Component, fp: &FootprintDef, net: NetId) -> bool {
    c.nets
        .iter()
        .enumerate()
        .any(|(pad_idx, pad_net)| *pad_net == Some(net) && !fp.pads[pad_idx].power_pin)
}

pub(crate) fn non_power_signal_net_count(c: &Component, fp: &FootprintDef) -> usize {
    c.nets
        .iter()
        .enumerate()
        .filter_map(|(pad_idx, net)| (!fp.pads[pad_idx].power_pin).then_some(*net).flatten())
        .collect::<BTreeSet<_>>()
        .len()
}

/// Index (0=left, 1=right, 2=bottom, 3=top) of the board edge closest to `p`.
#[inline]
fn nearest_edge_idx(p: Point, width: f64, height: f64) -> usize {
    let d = [p.x.to_mm(), width - p.x.to_mm(), p.y.to_mm(), height - p.y.to_mm()];
    d.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

pub(crate) fn nearest_board_edge_point(p: Point, width: f64, height: f64) -> Point {
    match nearest_edge_idx(p, width, height) {
        0 => Point::new(Nm::from_mm(0.0), p.y),
        1 => Point::new(Nm::from_mm(width), p.y),
        2 => Point::new(p.x, Nm::from_mm(0.0)),
        _ => Point::new(p.x, Nm::from_mm(height)),
    }
}

pub(crate) fn connector_ingress_unit(p: Point, width: f64, height: f64) -> (f64, f64) {
    match nearest_edge_idx(p, width, height) {
        0 => (1.0, 0.0),
        1 => (-1.0, 0.0),
        2 => (0.0, 1.0),
        _ => (0.0, -1.0),
    }
}
