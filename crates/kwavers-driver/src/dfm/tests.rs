//! Tests for the `dfm` slice (Phase 4l carve-out). Moved verbatim from the flat `src/dfm.rs`
//! `mod tests` block; `super::*` resolves the slice facade.

use super::*;
use crate::board::{Board, LayerId, NetClassKind, NetId, Track, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::rules::DesignRules;

fn board() -> Board {
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    Board::new(spec)
}

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

fn seg(b: &mut Board, net: NetId, x0: f64, y0: f64, x1: f64, y1: f64) {
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(x0), Nm::from_mm(y0)),
        end: Point::new(Nm::from_mm(x1), Nm::from_mm(y1)),
        width: Nm::from_mm(0.25),
        layer: LayerId(0),
        net,
    });
}

/// Total copper length on a layer — the invariant `merge_collinear` must preserve exactly.
fn total_len(b: &Board) -> f64 {
    b.tracks.iter().map(|t| t.start.euclid(t.end)).sum()
}

#[test]
fn ampacity_widening_respects_clearance_and_planes() {
    use crate::board::NetClassKind;
    let mut b = board(); // 20×20, 2 layers
    let hi = b.add_net("PWR", NetClassKind::Power);
    // An isolated power track with lots of room → widens up to the IPC-2221 requirement.
    seg(&mut b, hi, 2.0, 10.0, 18.0, 10.0);
    let removed = widen_for_ampacity(
        &mut b,
        |_| 1.0,
        10.0,
        1.0,
        Nm::from_mm(0.13),
        Nm::from_mm(0.5),
    );
    assert_eq!(removed, 1, "the open power track widens");
    let need = crate::physics::ampacity::ipc2221_min_width(1.0, 10.0, 1.0, true).to_mm();
    assert!(
        (b.tracks[0].width.to_mm() - need).abs() < 0.02,
        "widened to the ampacity requirement ({:.3} vs {:.3})",
        b.tracks[0].width.to_mm(),
        need
    );

    // A second power net hemmed in by a close foreign track cannot widen past clearance.
    let mut b2 = board();
    let a = b2.add_net("PWR", NetClassKind::Power);
    let foreign = b2.add_net("OTHER", NetClassKind::Signal);
    seg(&mut b2, a, 2.0, 10.0, 18.0, 10.0);
    seg(&mut b2, foreign, 2.0, 10.4, 18.0, 10.4); // 0.4 mm away, parallel
    widen_for_ampacity(
        &mut b2,
        |n| if n == a { 1.0 } else { 0.0 },
        10.0,
        1.0,
        Nm::from_mm(0.13),
        Nm::from_mm(0.5),
    );
    let w = b2.tracks.iter().find(|t| t.net == a).unwrap().width.to_mm();
    // Edge-to-edge to the foreign 0.25 mm track: gap 0.4 − 0.125(this min) − 0.125(other) must
    // stay ≥ 0.13 ⇒ this half ≤ 0.4 − 0.125 − 0.13 = 0.145 ⇒ width ≤ ~0.29 mm, far below the
    // ~0.42 mm requirement.
    assert!(
        (0.25..0.30).contains(&w),
        "hemmed-in track widens only within clearance (got {w:.3})"
    );
}

#[test]
fn ampacity_widening_respects_board_edge_clearance() {
    let mut b = board();
    let pwr = b.add_net("PWR", NetClassKind::Power);
    seg(&mut b, pwr, 2.0, 0.75, 18.0, 0.75);
    widen_for_ampacity(
        &mut b,
        |_| 1.0,
        10.0,
        1.0,
        Nm::from_mm(0.13),
        Nm::from_mm(0.5),
    );
    assert!(
        b.tracks[0].width.to_mm() <= 0.5 + 1.0e-6,
        "track center 0.75 mm from the edge may widen only to 0.5 mm width"
    );
}

#[test]
fn dedup_vias_merges_coincident_same_net_vias() {
    use crate::board::{NetClassKind, Via};
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
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

#[test]
fn ground_pour_falls_back_to_rectangle_without_features() {
    let mut b = board();
    let g = b.add_net("GND", NetClassKind::Ground);
    // No GND pads/vias ⇒ no hull ⇒ the inset board rectangle.
    assert!(ground_pour(&mut b, g, Nm::from_mm(1.0), LayerId(1)));
    assert_eq!(b.zones.len(), 1);
    let z = &b.zones[0];
    assert_eq!(z.net, g);
    assert_eq!(z.layer, LayerId(1));
    assert_eq!(z.polygon.len(), 4, "fallback is the inset board rectangle");
    assert!(matches!(z.fill, ZoneFill::ThermalRelief));
}

#[test]
fn ground_pour_bounds_to_the_ground_feature_hull() {
    use crate::board::Pad;
    let mut b = board();
    let g = b.add_net("GND", NetClassKind::Ground);
    // Four GND pads forming a small square in one quadrant; the pour hull must be that square,
    // not the whole board (so far-edge regions are never poured).
    for (x, y) in [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)] {
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            layers: vec![LayerId(0)],
            net: Some(g),
        });
    }
    assert!(ground_pour(&mut b, g, Nm::from_mm(1.0), LayerId(1)));
    let z = &b.zones[0];
    assert_eq!(z.polygon.len(), 4, "hull of the 4 GND pads is a quad");
    let maxx = z.polygon.iter().map(|p| p.x.0).max().unwrap();
    // The hull is inflated ~3 mm so boundary pads get thermal spokes, but the plane still hugs
    // the ground region (pads span x∈[3,7]) — nowhere near the 20 mm board edge.
    assert!(
        maxx <= Nm::from_mm(11.0).0,
        "pour stays near the ground features (bounded inflation, no full-board flood)"
    );
}

#[test]
fn teardrops_fillet_a_track_meeting_a_via() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    // A via with a track leaving it on layer 0 ⇒ one teardrop on that layer.
    let via_pos = Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0));
    b.vias.push(crate::board::Via {
        pos: via_pos,
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: n,
        from: LayerId(0),
        to: LayerId(1),
        kind: crate::board::ViaKind::Blind,
        filled: false,
    });
    seg(&mut b, n, 10.0, 10.0, 14.0, 10.0); // track from the via
    let added = teardrops(&mut b);
    assert_eq!(added, 1, "one track meets the via ⇒ one teardrop");
    let z = &b.zones[0];
    assert!(matches!(z.fill, ZoneFill::Solid));
    assert_eq!(z.net, n);
    assert_eq!(z.layer, LayerId(0));
    // A foreign-net via gets no teardrop.
    let m = b.add_net("M", NetClassKind::Signal);
    b.zones.clear();
    seg(&mut b, m, 0.0, 0.0, 4.0, 0.0); // far from the via, different net
    assert_eq!(
        teardrops(&mut b),
        1,
        "still only the same-net track is filleted"
    );
}

#[test]
fn chamfer_diagonal_traps_resolves_acute_angle_junctions() {
    // Steiner branch: from junction (1,1) one arm goes NE (diagonal) to (2,2),
    // another goes East (axial) to (2,1). These form a 45° acute angle at (1,1) — an acid trap.
    // After chamfering and collinear-merge the diagonal becomes East→North and all
    // junctions measure ≥ 90°.
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    // Diagonal arm: (1,1) → (2,2) [NE, 45°].
    seg(&mut b, n, 1.0, 1.0, 2.0, 2.0);
    // Axial arm:    (1,1) → (2,1) [East, 0°].
    seg(&mut b, n, 1.0, 1.0, 2.0, 1.0);

    let replaced = chamfer_diagonal_traps(&mut b);
    assert_eq!(replaced, 1, "one diagonal must be chamfered");
    assert_eq!(
            b.tracks.len(),
            3,
            "one diagonal (removed) becomes two orthogonal legs, plus the kept axial = 3 tracks before merge"
        );

    // merge_collinear removes the duplicate East leg the chamfer creates when the L-corner
    // lands on the endpoint of the existing axial segment going the same direction.
    merge_collinear(&mut b);

    // After chamfer + merge, no track pair sharing an endpoint should form an acute angle.
    let key = |p: Point| (p.x.0, p.y.0);
    let tr = &b.tracks;
    for i in 0..tr.len() {
        for j in (i + 1)..tr.len() {
            if tr[i].net != tr[j].net || tr[i].layer != tr[j].layer {
                continue;
            }
            let (a, b_pt) = (tr[i].start, tr[i].end);
            let (c, d) = (tr[j].start, tr[j].end);
            let maybe = if key(a) == key(c) {
                Some((a, b_pt, d))
            } else if key(a) == key(d) {
                Some((a, b_pt, c))
            } else if key(b_pt) == key(c) {
                Some((b_pt, a, d))
            } else if key(b_pt) == key(d) {
                Some((b_pt, a, c))
            } else {
                None
            };
            if let Some((apex, u, v)) = maybe {
                let ux = (u.x.0 - apex.x.0) as f64;
                let uy = (u.y.0 - apex.y.0) as f64;
                let vx = (v.x.0 - apex.x.0) as f64;
                let vy = (v.y.0 - apex.y.0) as f64;
                assert!(
                    ux * vx + uy * vy <= 0.0,
                    "acid trap remains after chamfer + merge at apex ({}, {})",
                    apex.x.0,
                    apex.y.0
                );
            }
        }
    }
}

#[test]
fn chamfer_diagonal_traps_preserves_clean_diagonal() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    seg(&mut b, n, 1.0, 1.0, 4.0, 4.0);

    let replaced = chamfer_diagonal_traps(&mut b);

    assert_eq!(replaced, 0, "isolated diagonal has no acute DFM trap");
    assert_eq!(b.tracks.len(), 1, "clean diagonal must remain one segment");
    assert_eq!(
        b.tracks[0].start,
        Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0))
    );
    assert_eq!(
        b.tracks[0].end,
        Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0))
    );
}

#[test]
fn plane_distribute_removes_tracks_and_stitches_only_off_plane_pads() {
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let sig = b.add_net("SIG", NetClassKind::Signal);
    // Two GND routed tracks (superseded by the plane) and one foreign signal track (kept).
    seg(&mut b, gnd, 2.0, 2.0, 5.0, 2.0);
    seg(&mut b, gnd, 5.0, 2.0, 5.0, 6.0);
    seg(&mut b, sig, 8.0, 8.0, 9.0, 8.0);
    // A GND via (superseded) and a foreign via (kept).
    let via = |net, from, to, x: f64| crate::board::Via {
        pos: Point::new(Nm::from_mm(x), Nm::from_mm(3.0)),
        drill: Nm::from_mm(0.3),
        diameter: Nm::from_mm(0.6),
        net,
        from: LayerId(from),
        to: LayerId(to),
        kind: crate::board::ViaKind::Through,
        filled: false,
    };
    b.vias.push(via(gnd, 0, 3, 4.0));
    b.vias.push(via(sig, 0, 1, 12.0));
    // GND pads: one SMD on F.Cu (needs a stitch to the inner plane), one thru-hole barrel
    // already reaching the plane (no stitch), and a foreign signal pad (untouched).
    b.pads.push(crate::board::Pad {
        pos: Point::new(Nm::from_mm(3.0), Nm::from_mm(10.0)),
        layers: vec![LayerId(0)],
        net: Some(gnd),
    });
    b.pads.push(crate::board::Pad {
        pos: Point::new(Nm::from_mm(6.0), Nm::from_mm(10.0)),
        layers: vec![LayerId(0), LayerId(1), LayerId(2), LayerId(3)],
        net: Some(gnd),
    });
    b.pads.push(crate::board::Pad {
        pos: Point::new(Nm::from_mm(9.0), Nm::from_mm(10.0)),
        layers: vec![LayerId(0)],
        net: Some(sig),
    });

    let plane = LayerId(1);
    let (removed, added) = plane_distribute_net(&mut b, gnd, &[plane], &DesignRules::holohv());

    assert_eq!(
        removed, 2,
        "both GND tracks removed (the plane carries them)"
    );
    assert_eq!(
        added, 1,
        "only the off-plane SMD GND pad gets a stitching via"
    );
    // The signal track survives; no GND track remains.
    assert_eq!(b.tracks.len(), 1);
    assert_eq!(b.tracks[0].net, sig);
    // GND vias = the single stitch (the original GND via was removed); the signal via survives.
    let gnd_vias: Vec<_> = b.vias.iter().filter(|v| v.net == gnd).collect();
    assert_eq!(gnd_vias.len(), 1, "exactly one stitching via for GND");
    assert!(
        gnd_vias[0].from.0 <= plane.0 && plane.0 <= gnd_vias[0].to.0,
        "stitch via barrel spans the plane layer (L{}..L{} covers L{})",
        gnd_vias[0].from.0,
        gnd_vias[0].to.0,
        plane.0
    );
    assert!(
        (gnd_vias[0].pos.x.to_mm() - 3.0).abs() < 1e-6,
        "stitch sits at the off-plane pad"
    );
    assert!(b.vias.iter().any(|v| v.net == sig), "foreign via untouched");
}

#[test]
fn resolve_diagonal_via_clearance_converts_violating_diagonal() {
    // Reproduce the FPGA board violation: DONE diagonal (50.5,39.5)→(51.0,40.0) on F.Cu
    // clears the PROG blind via at (51.0,39.5) by only 0.0486 mm < 0.13 mm min_clearance.
    use crate::board::ViaKind;
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(50.0), Nm::from_mm(0.5), 3).unwrap();
    let mut b = Board::new(spec);
    let net_done = b.add_net("DONE", NetClassKind::Signal);
    let net_prog = b.add_net("PROG", NetClassKind::Signal);
    let rules = DesignRules::holohv();

    // DONE diagonal: (50.5,39.5)→(51.0,40.0) on F.Cu (layer 0).
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(50.5), Nm::from_mm(39.5)),
        end: Point::new(Nm::from_mm(51.0), Nm::from_mm(40.0)),
        width: rules.signal_track,
        layer: LayerId(0),
        net: net_done,
    });
    // PROG blind via at (51.0,39.5) with 0.46 mm outer diameter (F.Cu→In2.Cu).
    b.vias.push(crate::board::Via {
        pos: Point::new(Nm::from_mm(51.0), Nm::from_mm(39.5)),
        drill: Nm::from_mm(0.2),
        diameter: Nm::from_mm(0.46),
        net: net_prog,
        from: LayerId(0),
        to: LayerId(2),
        kind: ViaKind::Blind,
        filled: false,
    });

    let converted = resolve_diagonal_via_clearance(&mut b, &rules);
    assert_eq!(
        converted, 1,
        "the DONE diagonal violates clearance to PROG via and must be converted"
    );
    // No diagonal remains for DONE on F.Cu.
    let diag_count = b
        .tracks
        .iter()
        .filter(|t| {
            let dx = (t.end.x.0 - t.start.x.0).abs();
            let dy = (t.end.y.0 - t.start.y.0).abs();
            t.net == net_done && t.layer == LayerId(0) && dx == dy && dx != 0
        })
        .count();
    assert_eq!(
        diag_count, 0,
        "all violating DONE diagonals converted to orthogonal"
    );

    // Verify every resulting segment clears the PROG via by ≥ min_clearance.
    use crate::geom::dist_point_seg;
    let via_pos = Point::new(Nm::from_mm(51.0), Nm::from_mm(39.5));
    let via_r = Nm::from_mm(0.23).0 as f64; // half of 0.46 mm diameter
    let mc = rules.min_clearance.0 as f64;
    for t in b
        .tracks
        .iter()
        .filter(|t| t.net == net_done && t.layer == LayerId(0))
    {
        let hw = t.width.0 as f64 / 2.0;
        let d = dist_point_seg(via_pos, t.start, t.end);
        let clearance = d - hw - via_r;
        assert!(
            clearance >= mc - 1.0, // 1 nm tolerance for integer rounding
            "segment ({:.3},{:.3})→({:.3},{:.3}) clearance {:.0} nm < min {:.0} nm",
            t.start.x.to_mm(),
            t.start.y.to_mm(),
            t.end.x.to_mm(),
            t.end.y.to_mm(),
            clearance,
            mc
        );
    }
}

#[test]
fn miter_right_angle_corners_creates_135_degree_bends() {
    // L-corner: horizontal (0,5)→(5,5) + vertical (5,5)→(5,0).
    // Apex at (5,5). After mitering with chamfer=0.5 mm:
    //   P1 = (4.5, 5.0) on the H arm (heading west from apex)
    //   P2 = (5.0, 4.5) on the V arm (heading south from apex)
    // Three tracks: H-stub (0,5)→(4.5,5), diagonal (4.5,5)→(5,4.5), V-stub (5,4.5)→(5,0).
    // Both new junctions subtend 135° → not flagged by detect_sharp_bends.
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b = Board::new(spec);
    let net = b.add_net("SIG", NetClassKind::Signal);
    let w = Nm::from_mm(0.15);
    let chamfer = Nm::from_mm(0.5);

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(0.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        width: w,
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(0.0)),
        width: w,
        layer: LayerId(0),
        net,
    });

    let mitered = miter_right_angle_corners(&mut b, chamfer, &[], &[], &DesignRules::holohv());
    assert_eq!(mitered, 1, "one right-angle junction must be mitered");
    assert_eq!(b.tracks.len(), 3, "H-stub + diagonal + V-stub");

    // Verify no sharp bends remain (all junction angles > 90°).
    let key = |p: Point| (p.x.0, p.y.0);
    let tr = &b.tracks;
    let mut sharp = 0usize;
    for i in 0..tr.len() {
        for j in (i + 1)..tr.len() {
            if tr[i].net != tr[j].net || tr[i].layer != tr[j].layer {
                continue;
            }
            let (a, b_pt) = (tr[i].start, tr[i].end);
            let (c, d) = (tr[j].start, tr[j].end);
            let maybe = if key(a) == key(c) {
                Some((a, b_pt, d))
            } else if key(a) == key(d) {
                Some((a, b_pt, c))
            } else if key(b_pt) == key(c) {
                Some((b_pt, a, d))
            } else if key(b_pt) == key(d) {
                Some((b_pt, a, c))
            } else {
                None
            };
            if let Some((apex, u, v)) = maybe {
                let ux = (u.x.0 - apex.x.0) as f64;
                let uy = (u.y.0 - apex.y.0) as f64;
                let vx = (v.x.0 - apex.x.0) as f64;
                let vy = (v.y.0 - apex.y.0) as f64;
                let u_len = ux.hypot(uy);
                let v_len = vx.hypot(vy);
                if u_len > f64::EPSILON && v_len > f64::EPSILON {
                    let dot = ux * vx + uy * vy;
                    if dot >= -1e-5 * u_len * v_len {
                        sharp += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        sharp, 0,
        "after mitering, no right-angle or acute junctions remain (found {sharp})"
    );

    // Verify the diagonal is exactly 45° (|dx| == |dy|).
    let diag = b
        .tracks
        .iter()
        .find(|t| {
            let dx = (t.end.x.0 - t.start.x.0).abs();
            let dy = (t.end.y.0 - t.start.y.0).abs();
            dx == dy && dx != 0
        })
        .expect("miter must produce a 45° diagonal chamfer");
    assert_eq!(
        diag.start,
        Point::new(Nm::from_mm(4.5), Nm::from_mm(5.0)),
        "P1 is chamfer distance from apex along H arm"
    );
    assert_eq!(
        diag.end,
        Point::new(Nm::from_mm(5.0), Nm::from_mm(4.5)),
        "P2 is chamfer distance from apex along V arm"
    );
}

#[test]
fn miter_right_angle_corners_patches_both_ends_of_shared_track() {
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b = Board::new(spec);
    let net = b.add_net("SIG", NetClassKind::Signal);
    let w = Nm::from_mm(0.15);

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
        width: w,
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(1.0), Nm::from_mm(4.0)),
        width: w,
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(4.0)),
        width: w,
        layer: LayerId(0),
        net,
    });

    let mitered =
        miter_right_angle_corners(&mut b, Nm::from_mm(0.5), &[], &[], &DesignRules::holohv());
    assert_eq!(mitered, 2, "both right-angle junctions are eligible");

    let mut degree = std::collections::HashMap::<(i64, i64), usize>::new();
    for t in &b.tracks {
        *degree.entry((t.start.x.0, t.start.y.0)).or_default() += 1;
        *degree.entry((t.end.x.0, t.end.y.0)).or_default() += 1;
    }

    let diagonal_junctions: Vec<Point> = b
        .tracks
        .iter()
        .filter(|t| {
            let dx = (t.end.x.0 - t.start.x.0).abs();
            let dy = (t.end.y.0 - t.start.y.0).abs();
            dx == dy && dx != 0
        })
        .flat_map(|t| [t.start, t.end])
        .collect();
    assert_eq!(
        diagonal_junctions.len(),
        4,
        "two 45-degree chamfers produce four explicit junction points"
    );
    for p in diagonal_junctions {
        assert_eq!(
            degree[&(p.x.0, p.y.0)],
            2,
            "each chamfer endpoint must be a split track endpoint"
        );
    }
}

#[test]
fn miter_skips_corner_too_close_to_foreign_pad() {
    // L-corner: H (0,5)→(5,5) + V (5,5)→(5,0). Apex (5,5), chamfer=0.5 mm.
    // P1=(4.5,5.0), P2=(5.0,4.5). Miter diagonal: P1→P2.
    //
    // Foreign pad centred at (4.5,4.6) mm, size 0.35×0.35 mm:
    //   half_diag ≈ 0.247 mm
    //   guard     = min_clearance(0.13) + track_half(0.075) = 0.205 mm
    //   min_dist  = 0.452 mm
    //   dist_point_seg((4.5,4.6), P1, P2) ≈ 0.283 mm  <  0.452 mm  → SKIP
    use crate::place::component::{Component, Placement};
    use crate::place::footprint::{FootprintDef, PadDef, Role};
    use crate::place::rotation::Rot;

    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 1).unwrap();
    let mut b = Board::new(spec);
    let sig = b.add_net("SIG", NetClassKind::Signal);
    let fgn = b.add_net("FOREIGN", NetClassKind::Signal);
    let w = Nm::from_mm(0.15);

    // Horizontal arm
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(0.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        width: w,
        layer: LayerId(0),
        net: sig,
    });
    // Vertical arm
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(0.0)),
        width: w,
        layer: LayerId(0),
        net: sig,
    });
    // One-pad passive footprint: pad at centre, size 0.35×0.35 mm.
    let lib = vec![FootprintDef::new(
        "SMD_PAD",
        (Nm::from_mm(1.0), Nm::from_mm(1.0)),
        Role::Passive,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.35), Nm::from_mm(0.35)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    // Component placed at (4.5, 4.6): pad is at that position (offset = 0).
    let comps = vec![Component {
        fp: 0,
        nets: vec![Some(fgn)],
        refdes: "R1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(4.5), Nm::from_mm(4.6)),
            rot: Rot::R0,
        },
        ..Default::default()
    }];
    let rules = DesignRules::holohv();
    let mitered = miter_right_angle_corners(&mut b, Nm::from_mm(0.5), &comps, &lib, &rules);
    assert_eq!(
        mitered, 0,
        "miter must be skipped: diagonal P1→P2 is {:.3} mm from foreign pad edge, \
             threshold {:.3} mm",
        0.283_f64, 0.452_f64
    );
    assert_eq!(b.tracks.len(), 2, "original L-corner tracks unchanged");
}

#[test]
fn split_track_body_junctions_creates_explicit_endpoint() {
    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b = Board::new(spec);
    let net = b.add_net("SIG", NetClassKind::Signal);
    let w = Nm::from_mm(0.15);
    let junction = Point::new(Nm::from_mm(3.0), Nm::from_mm(1.0));

    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(5.0), Nm::from_mm(1.0)),
        width: w,
        layer: LayerId(0),
        net,
    });
    b.tracks.push(Track {
        start: junction,
        end: Point::new(Nm::from_mm(3.0), Nm::from_mm(4.0)),
        width: w,
        layer: LayerId(0),
        net,
    });

    let split = split_track_body_junctions(&mut b);
    assert_eq!(split, 1, "the horizontal body segment is split once");
    assert_eq!(b.tracks.len(), 3, "two horizontal spans plus the branch");

    let degree = b
        .tracks
        .iter()
        .flat_map(|t| [t.start, t.end])
        .filter(|&p| p == junction)
        .count();
    assert_eq!(
        degree, 3,
        "the T-junction is represented by exact endpoints"
    );
}

#[test]
fn pad_entry_stubs_skips_stub_that_violates_adjacent_pad_clearance() {
    // Two pads on the same IC: A (net 0) and B (net 1) 0.42 mm apart.
    // B's stub would pass within 0.1 mm of A's copper edge — below 0.13 mm clearance.
    // The guard must skip B's stub, leaving stub count = 0 (A is on grid, no stub; B is skipped).
    use crate::board::{Board, LayerId, NetClassKind};
    use crate::geom::{GridSpec, Nm, Point};
    use crate::place::component::{Component, Placement};
    use crate::place::footprint::{FootprintDef, PadDef, Role};
    use crate::place::rotation::Rot;
    use crate::rules::DesignRules;

    let spec = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 1).unwrap();
    let mut board = Board::new(spec);
    let net_a = board.add_net("A", NetClassKind::Signal);
    let net_b = board.add_net("B", NetClassKind::Signal);

    // Pad A: on grid at (5.0, 5.0)}; no stub needed.
    // Pad B: off-grid at (5.0, 5.42) — only 0.42 mm from A. B's snapped node is (5.0, 5.5)
    // and the stub goes from (5.0, 5.5) to (5.0, 5.42), which passes 0.08 mm from A's edge
    // (A has size 0.55×0.55, half-diagonal = 0.389 mm}; 0.42 < 0.389 + 0.13 + 0.075 = 0.595 mm).
    let lib = vec![FootprintDef::new(
        "TWO_PAD",
        (Nm::from_mm(2.0), Nm::from_mm(2.0)),
        Role::ActiveIc,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
                size: (Nm::from_mm(0.55), Nm::from_mm(0.55)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.42)),
                size: (Nm::from_mm(0.35), Nm::from_mm(0.35)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )];
    let comps = vec![Component {
        fp: 0,
        nets: vec![Some(net_a), Some(net_b)],
        refdes: "U1".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }];
    let rules = DesignRules::holohv();
    // Add pads to board so board.class_of works.
    board.add_pad(crate::board::Pad {
        pos: comps[0].pad_pos(&lib, 0),
        layers: vec![LayerId(0)],
        net: Some(net_a),
    });
    board.add_pad(crate::board::Pad {
        pos: comps[0].pad_pos(&lib, 1),
        layers: vec![LayerId(0)],
        net: Some(net_b),
    });
    let added = pad_entry_stubs(&mut board, &comps, &lib, &rules);
    // Pad A is on grid (0.0 offset from IC at 5.0,5.0 → cell (10,10) → snapped == exact): 0 stubs.
    // Pad B would be off-grid but its stub violates clearance to A: 0 stubs.
    assert_eq!(
        added, 0,
        "stub that violates adjacent-pad clearance must be skipped"
    );
}
