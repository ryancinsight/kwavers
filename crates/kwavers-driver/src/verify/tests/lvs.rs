//! Family A — LVS tests.

use super::*;
use crate::board::{Board, LayerId, NetClassKind, Pad, Track};
use crate::geom::{GridSpec, Nm, Point};

#[test]
fn lvs_passes_when_a_track_joins_both_pads() {
    let s = spec();
    let mut b = Board::new(s);
    let a = b.add_net("A", NetClassKind::Signal);
    // Two pads connected by a track that walks the cells between them.
    let p0 = Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0));
    let p1 = Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0));
    b.add_pad(Pad {
        pos: p0,
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    b.add_pad(Pad {
        pos: p1,
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    b.tracks.push(Track {
        start: p0,
        end: p1,
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: a,
    });
    let r = lvs(&b);
    assert!(r.pass, "single net joined by a track is clean: {r:?}");
}

#[test]
fn lvs_connects_overlapping_same_net_tracks_on_distinct_cells() {
    // Two same-net collinear segments separated by a 0.3 mm centre gap but each 0.4 mm wide, so
    // their copper overlaps (edge gap −0.1 mm) — yet their abutting ends round to *different*
    // grid cells (6.0 mm → cell 12, 6.3 mm → cell 13 at 0.5 mm pitch). The cell union alone would
    // split the net into two islands (a false open); the geometric touch-union must join them, as
    // kicad-cli does. This is the dense-escape case (parallel diagonals the DFM leaves ~0.14 mm
    // apart) reduced to its essence.
    let s = spec();
    let mut b = Board::new(s);
    let a = b.add_net("A", NetClassKind::Signal);
    let p0 = Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0));
    let p1 = Point::new(Nm::from_mm(8.0), Nm::from_mm(4.0));
    b.add_pad(Pad {
        pos: p0,
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    b.add_pad(Pad {
        pos: p1,
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    let seg = |x0: f64, x1: f64, net| Track {
        start: Point::new(Nm::from_mm(x0), Nm::from_mm(4.0)),
        end: Point::new(Nm::from_mm(x1), Nm::from_mm(4.0)),
        width: Nm::from_mm(0.4),
        layer: LayerId(0),
        net,
    };
    b.tracks.push(seg(4.0, 6.0, a)); // from p0, ends at cell 12
    b.tracks.push(seg(6.3, 8.0, a)); // to p1, starts at cell 13 — copper overlaps the first
    assert!(
        lvs(&b).pass,
        "overlapping same-net copper on distinct cells must read as connected"
    );

    // Control: widen the gap past the copper so they genuinely do not touch ⇒ a real open.
    let mut b2 = Board::new(s);
    let a2 = b2.add_net("A", NetClassKind::Signal);
    b2.add_pad(Pad {
        pos: p0,
        layers: vec![LayerId(0)],
        net: Some(a2),
    });
    b2.add_pad(Pad {
        pos: p1,
        layers: vec![LayerId(0)],
        net: Some(a2),
    });
    b2.tracks.push(seg(4.0, 5.5, a2));
    b2.tracks.push(seg(7.0, 8.0, a2)); // 1.5 mm gap ≫ 0.4 mm copper ⇒ no touch
    assert!(
        !lvs(&b2).pass,
        "a genuine copper gap must still report an open"
    );
}

#[test]
fn lvs_reports_an_open_when_the_track_is_missing() {
    let s = spec();
    let mut b = Board::new(s);
    let a = b.add_net("A", NetClassKind::Signal);
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    b.add_pad(Pad {
        pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0)),
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    // No track ⇒ the two pads are separate islands ⇒ open.
    let r = lvs(&b);
    assert_eq!(r.opens, vec![("A".to_string(), 2)]);
    assert!(!r.pass);
}

#[test]
fn lvs_reports_a_short_when_copper_bridges_two_nets() {
    let s = spec();
    let mut b = Board::new(s);
    let a = b.add_net("A", NetClassKind::Signal);
    let c = b.add_net("B", NetClassKind::Signal);
    let p0 = Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0));
    let p1 = Point::new(Nm::from_mm(8.0), Nm::from_mm(10.0));
    b.add_pad(Pad {
        pos: p0,
        layers: vec![LayerId(0)],
        net: Some(a),
    });
    b.add_pad(Pad {
        pos: p1,
        layers: vec![LayerId(0)],
        net: Some(c),
    });
    // A track on net A that physically reaches B's pad ⇒ the two nets share an island ⇒ short.
    b.tracks.push(Track {
        start: p0,
        end: p1,
        width: Nm::from_mm(0.2),
        layer: LayerId(0),
        net: a,
    });
    let r = lvs(&b);
    assert_eq!(r.shorts, vec![("A".to_string(), "B".to_string())]);
    assert!(!r.pass);
}

#[test]
fn lvs_catches_a_thru_hole_barrel_short_on_an_inner_layer() {
    // The short kicad-cli caught but the internal check used to miss: a track on an *inner* layer
    // crossing a **thru-hole** pad of another net. The thru-hole pad is a full-stack barrel, so a
    // track touching its cell on any layer shorts to its net.
    let s = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(s);
    let g = b.add_net("GND", NetClassKind::Ground);
    let sig = b.add_net("SDI", NetClassKind::Signal);
    let pad = Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0));
    // Thru-hole GND pad (layers list is the [0,1] multi-layer marker → full barrel).
    b.add_pad(Pad {
        pos: pad,
        layers: vec![LayerId(0), LayerId(1)],
        net: Some(g),
    });
    // A separate SDI net with two pads on inner layer In2, and a track between them that runs
    // through the GND barrel's cell on In2.
    let s0 = Point::new(Nm::from_mm(6.0), Nm::from_mm(10.0));
    let s1 = Point::new(Nm::from_mm(14.0), Nm::from_mm(10.0));
    b.add_pad(Pad {
        pos: s0,
        layers: vec![LayerId(2)],
        net: Some(sig),
    });
    b.add_pad(Pad {
        pos: s1,
        layers: vec![LayerId(2)],
        net: Some(sig),
    });
    b.tracks.push(Track {
        start: s0,
        end: s1,
        width: Nm::from_mm(0.2),
        layer: LayerId(2), // inner layer — crosses the GND barrel at x=10
        net: sig,
    });
    let r = lvs(&b);
    assert_eq!(
        r.shorts,
        vec![("GND".to_string(), "SDI".to_string())],
        "an inner-layer track through a thru-hole barrel must register as a short"
    );
    assert!(!r.pass);
}
