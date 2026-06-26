//! Family B — power & plane tests.

use super::*;
use crate::board::{Board, LayerId, NetClassKind, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::rules::DesignRules;

#[test]
fn ampacity_widening_respects_clearance_and_planes() {
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
