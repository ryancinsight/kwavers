//! Family A — track cleanup tests.

use super::*;
use crate::board::{LayerId, NetClassKind};
use crate::geom::{Nm, Point};
use crate::rules::DesignRules;

#[test]
fn remove_orphan_copper_deletes_padless_islands_keeps_connected() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    // Connected copper: two pads joined by a track (an island that contains pads → kept).
    b.pads.push(crate::board::Pad {
        pos: Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0)),
        layers: vec![LayerId(0)],
        net: Some(n),
    });
    b.pads.push(crate::board::Pad {
        pos: Point::new(Nm::from_mm(8.0), Nm::from_mm(4.0)),
        layers: vec![LayerId(0)],
        net: Some(n),
    });
    seg(&mut b, n, 4.0, 4.0, 8.0, 4.0);
    // Orphan copper: a via + stub at (15,15) connected to no pad (an island with no pad → removed).
    b.vias.push(crate::board::Via {
        pos: Point::new(Nm::from_mm(15.0), Nm::from_mm(15.0)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.6),
        net: n,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Through,
        filled: false,
    });
    seg(&mut b, n, 15.0, 15.0, 15.0, 17.0);
    assert_eq!(b.tracks.len(), 2);
    assert_eq!(b.vias.len(), 1);
    let removed = remove_orphan_copper(&mut b);
    assert_eq!(removed, 2, "the orphan via + its stub are removed");
    assert_eq!(b.vias.len(), 0, "no via survives — the only one was orphan");
    // The pad-to-pad connection survives intact.
    assert_eq!(b.tracks.len(), 1);
    assert_eq!(
        b.tracks[0].start,
        Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0))
    );
}

#[test]
fn trim_dangling_removes_antenna_keeps_anchored_path() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    let via = |x: f64, y: f64| crate::board::Via {
        pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.6),
        net: n,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Through,
        filled: false,
    };
    // Two anchored vias joined by a two-segment path through a junction at (5,2), plus a stub
    // branching off that junction to a free end at (5,6) — the antenna.
    b.vias.push(via(2.0, 2.0));
    b.vias.push(via(8.0, 2.0));
    seg(&mut b, n, 2.0, 2.0, 5.0, 2.0); // A1: via → junction
    seg(&mut b, n, 5.0, 2.0, 8.0, 2.0); // A2: junction → via
    seg(&mut b, n, 5.0, 2.0, 5.0, 6.0); // B: junction → free end (dangling antenna)
    assert_eq!(b.tracks.len(), 3);
    let removed = trim_dangling_stubs(&mut b);
    assert_eq!(removed, 1, "only the antenna stub is removed");
    assert_eq!(b.tracks.len(), 2, "the via-to-via path is preserved");
    // No remaining segment touches the free end (5,6); both vias still bridged.
    let touches = |x: f64, y: f64| {
        b.tracks.iter().any(|t| {
            (t.start == Point::new(Nm::from_mm(x), Nm::from_mm(y)))
                || (t.end == Point::new(Nm::from_mm(x), Nm::from_mm(y)))
        })
    };
    assert!(!touches(5.0, 6.0), "antenna free end is gone");
    assert!(
        touches(2.0, 2.0) && touches(8.0, 2.0),
        "both anchors stay connected"
    );
}

#[test]
fn trim_dangling_peels_a_multi_segment_stub_to_the_junction() {
    // A stub two segments long off an anchored trunk must peel fully back in one call.
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    b.vias.push(crate::board::Via {
        pos: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.6),
        net: n,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Through,
        filled: false,
    });
    seg(&mut b, n, 2.0, 2.0, 5.0, 2.0); // anchored trunk to junction
    seg(&mut b, n, 5.0, 2.0, 7.0, 2.0); // anchored trunk continues (degree-1 end at 7,2 — but…)
                                        // Make (7,2) anchored too so the trunk is kept; the stub hangs off (5,2).
    b.vias.push(crate::board::Via {
        pos: Point::new(Nm::from_mm(7.0), Nm::from_mm(2.0)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.6),
        net: n,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Through,
        filled: false,
    });
    seg(&mut b, n, 5.0, 2.0, 5.0, 4.0); // stub seg 1
    seg(&mut b, n, 5.0, 4.0, 5.0, 7.0); // stub seg 2 → free end
    let removed = trim_dangling_stubs(&mut b);
    assert_eq!(removed, 2, "both stub segments peel back to the junction");
    assert_eq!(b.tracks.len(), 2, "the anchored trunk survives");
}

#[test]
fn dedup_vias_merges_coincident_same_net_vias() {
    use crate::board::{NetClassKind, Via};
    use crate::geom::GridSpec;
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = crate::board::Board::new(spec);
    let n = b.add_net("N", NetClassKind::Signal);
    let m = b.add_net("M", NetClassKind::Signal);
    let at = Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0));
    let via = |net, from, to| Via {
        pos: at,
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net,
        from: LayerId(from),
        to: LayerId(to),
        kind: crate::board::ViaKind::Through,
        filled: false,
    };
    // Two same-net vias at the same spot (a BGA fanout F.Cu→In1 + a routing via In1→B.Cu)…
    b.vias.push(via(n, 0, 1));
    b.vias.push(via(n, 1, 3));
    // …plus a different-net via at the same spot (must NOT merge with net N).
    b.vias.push(via(m, 0, 1));
    let removed = dedup_vias(&mut b, &DesignRules::default());
    assert_eq!(removed, 1, "the two net-N vias merge into one");
    assert_eq!(b.vias.len(), 2, "net N (merged) + net M");
    let nv = b.vias.iter().find(|v| v.net == n).unwrap();
    assert_eq!(
        (nv.from, nv.to),
        (LayerId(0), LayerId(3)),
        "spans the union of layers"
    );
}

#[test]
fn merges_a_straight_run_into_one_segment() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    // Four 0.5 mm abutting segments forming one 2 mm run.
    for i in 0..4 {
        let x = 2.0 + i as f64 * 0.5;
        seg(&mut b, n, x, 5.0, x + 0.5, 5.0);
    }
    let len0 = total_len(&b);
    let removed = merge_collinear(&mut b);
    assert_eq!(removed, 3, "four abutting segments collapse to one");
    assert_eq!(b.tracks.len(), 1);
    assert!(
        (total_len(&b) - len0).abs() < 1.0,
        "copper length is preserved exactly (nm rounding only)"
    );
    // The single segment spans the full run.
    assert_eq!(b.tracks[0].start.x, Nm::from_mm(2.0));
    assert_eq!(b.tracks[0].end.x, Nm::from_mm(4.0));
}

#[test]
fn keeps_corners_and_separates_nets_layers() {
    let mut b = board();
    let a = b.add_net("A", NetClassKind::Signal);
    let c = b.add_net("C", NetClassKind::Signal);
    // Net A: an L (horizontal run + vertical run) — corner must survive as two segments.
    seg(&mut b, a, 2.0, 5.0, 2.5, 5.0);
    seg(&mut b, a, 2.5, 5.0, 3.0, 5.0);
    seg(&mut b, a, 3.0, 5.0, 3.0, 5.5);
    seg(&mut b, a, 3.0, 5.5, 3.0, 6.0);
    // Net C overlapping A's row but a different net — never merged with A.
    seg(&mut b, c, 2.0, 5.0, 2.5, 5.0);
    let removed = merge_collinear(&mut b);
    assert_eq!(removed, 2, "each of A's two runs collapses 2→1; C stays");
    // A → 2 segments (horizontal + vertical), C → 1 segment.
    assert_eq!(b.tracks.len(), 3);
    let a_count = b.tracks.iter().filter(|t| t.net == a).count();
    assert_eq!(a_count, 2, "the 90° corner is preserved as two segments");
}

#[test]
fn quietest_layer_picks_the_least_routed() {
    let mut b = board(); // 2 layers
    let n = b.add_net("N", NetClassKind::Signal);
    // Heavy run on layer 0, nothing on layer 1 ⇒ layer 1 is quietest.
    seg(&mut b, n, 1.0, 1.0, 19.0, 1.0);
    assert_eq!(quietest_layer(&b), LayerId(1));
}
