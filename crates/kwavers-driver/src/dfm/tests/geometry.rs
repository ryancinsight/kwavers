//! Family C — geometric correction tests.

use super::*;
use crate::board::{Board, LayerId, NetClassKind, Track};
use crate::geom::{GridSpec, Nm, Point};
use crate::rules::DesignRules;

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
