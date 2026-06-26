//! Cost-seam property/differential tests.
//!
//! Tests live in their own file (gated `#[cfg(test)]`) so the production module tree stays
//! slim and the test surface is auditable as a single file. Test count is preserved from the
//! pre-split `src/cost.rs::mod tests`: the 12 tests below were moved verbatim with only one
//! import added — `use super::physics::HIGH_SPEED_VIA_MULTIPLIER` — because the per-class
//! penalty constants are `pub(super)` in `physics.rs` (scope-visible to `crate::cost` and its
//! children) rather than re-exported through `crate::cost`'s public `pub use` surface.

use super::physics::HIGH_SPEED_VIA_MULTIPLIER;
use super::*;

use crate::board::{Board, LayerId, NetClassKind, Pad, Track, Zone, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::rules::{CreepageRule, DesignRules};

fn board_with_lv_pad_at(p: Point) -> (Board, GridSpec) {
    let spec = GridSpec::cover(Nm::from_mm(5.0), Nm::from_mm(5.0), Nm::from_mm(0.25), 4).unwrap();
    let mut b = Board::new(spec);
    let lv = b.add_net("CTRL", NetClassKind::Signal);
    b.add_pad(Pad {
        pos: p,
        layers: vec![LayerId(0)],
        net: Some(lv),
    });
    (b, spec)
}

#[test]
fn proximity_accumulates_from_multiple_nearby_pads() {
    // Two LV pads flanking the test point, each within creep radius (0.5 mm), same distance
    // (0.25 mm). The sum model gives hazard = min(0.5 + 0.5, 1) = 1.0 whereas nearest-only
    // gives 0.5. The HV cost at the flanked cell must exceed the single-pad case.
    let spec = GridSpec::cover(Nm::from_mm(5.0), Nm::from_mm(5.0), Nm::from_mm(0.25), 4).unwrap();
    let p = Point::new(Nm::from_mm(2.5), Nm::from_mm(2.5));
    let pad = |x: f64| Pad {
        pos: Point::new(Nm::from_mm(x), Nm::from_mm(2.5)),
        layers: vec![LayerId(0)],
        net: None, // net assigned below
    };

    // Single LV pad 0.25 mm to the left.
    let mut b1 = Board::new(spec);
    let lv1 = b1.add_net("CTRL", NetClassKind::Signal);
    b1.add_pad(Pad {
        net: Some(lv1),
        ..pad(2.25)
    });
    let cost1 = PhysicsCost::new(
        spec,
        &b1,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        50.0,
        0.0,
    );

    // Two LV pads equidistant flanking the test point (0.25 mm each side).
    let mut b2 = Board::new(spec);
    let lv2 = b2.add_net("CTRL", NetClassKind::Signal);
    b2.add_pad(Pad {
        net: Some(lv2),
        ..pad(2.25)
    });
    b2.add_pad(Pad {
        net: Some(lv2),
        ..pad(2.75)
    });
    let cost2 = PhysicsCost::new(
        spec,
        &b2,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        50.0,
        0.0,
    );

    let hv_one = cost1.node_base(p, LayerId(0), NetClassKind::Hv);
    let hv_two = cost2.node_base(p, LayerId(0), NetClassKind::Hv);
    assert!(
        hv_two > hv_one,
        "two flanking LV pads must raise the HV routing cost more than a single pad: \
         one={hv_one:.3} two={hv_two:.3}"
    );
}

#[test]
fn hv_pays_more_near_a_low_voltage_pad() {
    let lv_at = Point::new(Nm::from_mm(2.5), Nm::from_mm(2.5));
    let (b, spec) = board_with_lv_pad_at(lv_at);
    let cost = PhysicsCost::new(
        spec,
        &b,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        50.0,
        1.0,
    );

    // A point right at the LV pad vs one far away, both for an HV net on an outer layer.
    let near = cost.node_base(lv_at, LayerId(0), NetClassKind::Hv);
    let far = cost.node_base(
        Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
        LayerId(0),
        NetClassKind::Hv,
    );
    assert!(
        near > far + 10.0,
        "HV cost near LV pad ({near}) must exceed the far cost ({far}) by the creepage weight"
    );
}

#[test]
fn hv_prefers_outer_layer() {
    let (b, spec) = board_with_lv_pad_at(Point::new(Nm::from_mm(2.5), Nm::from_mm(2.5)));
    let cost = PhysicsCost::new(
        spec,
        &b,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        50.0,
        5.0,
    );
    let corner = Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0));
    let outer = cost.node_base(corner, LayerId(0), NetClassKind::Hv);
    let inner = cost.node_base(corner, LayerId(1), NetClassKind::Hv);
    assert!(
        inner > outer,
        "HV on inner layer ({inner}) should cost more than outer ({outer})"
    );
}

#[test]
fn high_speed_classes_pay_extra_via_cost() {
    let (b, spec) = board_with_lv_pad_at(Point::new(Nm::from_mm(2.5), Nm::from_mm(2.5)));
    let cost = PhysicsCost::new(
        spec,
        &b,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );

    let signal = cost.via_cost(NetClassKind::Signal);
    let hv = cost.via_cost(NetClassKind::Hv);
    let power = cost.via_cost(NetClassKind::Power);
    let ground = cost.via_cost(NetClassKind::Ground);

    assert_eq!(
        signal,
        power * HIGH_SPEED_VIA_MULTIPLIER,
        "signal nets pay the guide-derived high-speed via penalty"
    );
    assert_eq!(
        hv,
        power * HIGH_SPEED_VIA_MULTIPLIER,
        "HV switching nets pay the same high-speed via penalty"
    );
    assert_eq!(
        ground, power,
        "plane-like return nets keep the base via cost"
    );
}

#[test]
fn high_speed_cost_prefers_top_outer_layer_over_bottom() {
    let (b, spec) = board_with_lv_pad_at(Point::new(Nm::from_mm(2.5), Nm::from_mm(2.5)));
    let cost = PhysicsCost::new(
        spec,
        &b,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        4.0,
    );
    let p = Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0));

    let top_signal = cost.node_base(p, LayerId(0), NetClassKind::Signal);
    let bottom_signal = cost.node_base(p, LayerId(3), NetClassKind::Signal);
    let top_hv = cost.node_base(p, LayerId(0), NetClassKind::Hv);
    let bottom_hv = cost.node_base(p, LayerId(3), NetClassKind::Hv);
    let top_ground = cost.node_base(p, LayerId(0), NetClassKind::Ground);
    let bottom_ground = cost.node_base(p, LayerId(3), NetClassKind::Ground);

    assert!(
        bottom_signal > top_signal,
        "high-speed signal routing should prefer top-side copper over bottom-side copper"
    );
    assert!(
        bottom_hv > top_hv,
        "HV switching routing should prefer top-side copper over bottom-side copper"
    );
    assert_eq!(
        bottom_ground, top_ground,
        "plane-like ground routing does not pay the top-side high-speed preference"
    );
}

#[test]
fn high_speed_signal_cost_prefers_adjacent_reference_plane() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(6.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let gnd = board.add_net("GND", NetClassKind::Ground);
    board.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(4.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Point::new(Nm::from_mm(1.0), Nm::from_mm(4.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );
    let referenced = cost.node_base(
        Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let unreferenced = cost.node_base(
        Point::new(Nm::from_mm(6.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    assert!(
        unreferenced > referenced,
        "high-speed signal cost must prefer cells covered by an adjacent ground/power reference plane"
    );

    let power_unreferenced = cost.node_base(
        Point::new(Nm::from_mm(6.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Power,
    );
    assert_eq!(
        power_unreferenced, 1.0,
        "plane-like power routing does not pay the high-speed signal reference-plane penalty"
    );
}

#[test]
fn high_speed_signal_cost_prefers_ground_reference_over_power_reference() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(6.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let pwr = board.add_net("VDD", NetClassKind::Power);
    let gnd = board.add_net("GND", NetClassKind::Ground);
    let zone = |net, x0, x1| Zone {
        net,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(x0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(4.0)),
            Point::new(Nm::from_mm(x0), Nm::from_mm(4.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };
    board.zones.push(zone(pwr, 1.0, 3.0));
    board.zones.push(zone(gnd, 4.0, 6.0));

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );
    let power_referenced = cost.node_base(
        Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let ground_referenced = cost.node_base(
        Point::new(Nm::from_mm(5.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let unreferenced = cost.node_base(
        Point::new(Nm::from_mm(7.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );

    assert!(
        unreferenced > power_referenced,
        "a power-plane reference is still better than no adjacent reference plane"
    );
    assert!(
        power_referenced > ground_referenced,
        "ground-backed routing should cost less than power-backed routing because power references require stitching capacitors"
    );

    let power_net_cost = cost.node_base(
        Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Power,
    );
    assert_eq!(
        power_net_cost, 1.0,
        "plane-like power routing does not pay the high-speed power-reference penalty"
    );
}

#[test]
fn high_speed_signal_cost_prefers_reference_plane_margin() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(6.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let gnd = board.add_net("GND", NetClassKind::Ground);
    board.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(6.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(6.0), Nm::from_mm(5.0)),
            Point::new(Nm::from_mm(1.0), Nm::from_mm(5.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });

    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );
    let weak_margin = cost.node_base(
        Point::new(Nm::from_mm(1.1), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let strong_margin = cost.node_base(
        Point::new(Nm::from_mm(3.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let unreferenced = cost.node_base(
        Point::new(Nm::from_mm(7.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );

    assert!(
        weak_margin > strong_margin,
        "a high-speed node near the reference-zone boundary should cost more than one with stronger plane margin"
    );
    assert!(
        unreferenced > weak_margin,
        "no adjacent reference plane remains worse than a referenced node with weak margin"
    );
}

#[test]
fn high_speed_signal_cost_prefers_board_edge_clearance() {
    let spec = GridSpec::cover(Nm::from_mm(6.0), Nm::from_mm(4.0), Nm::from_mm(0.25), 4).unwrap();
    let board = Board::new(spec);
    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );

    let near_edge = cost.node_base(
        Point::new(Nm::from_mm(0.25), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let interior = cost.node_base(
        Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    assert!(
        near_edge > interior,
        "high-speed signal routing near the board edge should cost more than interior routing"
    );

    let near_edge_power = cost.node_base(
        Point::new(Nm::from_mm(0.25), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Power,
    );
    let interior_power = cost.node_base(
        Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        LayerId(0),
        NetClassKind::Power,
    );
    assert_eq!(
        near_edge_power, interior_power,
        "plane-like power routing does not pay the high-speed edge penalty"
    );
}

#[test]
fn high_speed_signal_cost_prefers_extra_spacing_from_existing_high_speed_copper() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(6.0), Nm::from_mm(0.25), 4).unwrap();
    let mut board = Board::new(spec);
    let routed = board.add_net("TX_EXISTING", NetClassKind::Signal);
    board.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(3.0)),
        end: Point::new(Nm::from_mm(7.0), Nm::from_mm(3.0)),
        width: DesignRules::holohv().signal_track,
        layer: LayerId(0),
        net: routed,
    });
    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );

    let near = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(3.5)),
        LayerId(0),
        NetClassKind::Signal,
    );
    let far = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
        LayerId(0),
        NetClassKind::Signal,
    );
    assert!(
        near > far,
        "high-speed routing should prefer extra spacing from existing high-speed copper"
    );

    let near_power = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(3.5)),
        LayerId(0),
        NetClassKind::Power,
    );
    let far_power = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
        LayerId(0),
        NetClassKind::Power,
    );
    assert_eq!(
        near_power, far_power,
        "plane-like power routing does not pay the high-speed trace-spacing preference"
    );
}

#[test]
fn high_speed_signal_cost_prefers_adjacent_layer_lateral_separation() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(6.0), Nm::from_mm(0.25), 4).unwrap();
    let mut board = Board::new(spec);
    let routed = board.add_net("TX_EXISTING", NetClassKind::Signal);
    board.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(3.0)),
        end: Point::new(Nm::from_mm(7.0), Nm::from_mm(3.0)),
        width: DesignRules::holohv().signal_track,
        layer: LayerId(0),
        net: routed,
    });
    let cost = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    );

    let broadside = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(3.0)),
        LayerId(1),
        NetClassKind::Signal,
    );
    let separated = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
        LayerId(1),
        NetClassKind::Signal,
    );
    assert!(
        broadside > separated,
        "high-speed routing should avoid broadside parallel overlap on adjacent layers"
    );

    let broadside_power = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(3.0)),
        LayerId(1),
        NetClassKind::Power,
    );
    let separated_power = cost.node_base(
        Point::new(Nm::from_mm(4.0), Nm::from_mm(5.0)),
        LayerId(1),
        NetClassKind::Power,
    );
    assert_eq!(
        broadside_power, separated_power,
        "plane-like power routing does not pay adjacent-layer high-speed broadside cost"
    );
}

#[test]
fn inner_high_speed_signal_cost_prefers_dual_adjacent_ground_planes() {
    let spec = GridSpec::cover(Nm::from_mm(8.0), Nm::from_mm(6.0), Nm::from_mm(0.5), 4).unwrap();
    let mut board = Board::new(spec);
    let gnd = board.add_net("GND", NetClassKind::Ground);
    let zone = |layer| Zone {
        net: gnd,
        layer,
        polygon: vec![
            Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(4.0), Nm::from_mm(1.0)),
            Point::new(Nm::from_mm(4.0), Nm::from_mm(4.0)),
            Point::new(Nm::from_mm(1.0), Nm::from_mm(4.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };
    let p = Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0));

    let no_reference = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    )
    .node_base(p, LayerId(1), NetClassKind::Signal);

    board.zones.push(zone(LayerId(0)));
    let one_sided = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    )
    .node_base(p, LayerId(1), NetClassKind::Signal);

    board.zones.push(zone(LayerId(2)));
    let dual_ground = PhysicsCost::new(
        spec,
        &board,
        &DesignRules::holohv(),
        CreepageRule::holohv(),
        0.0,
        0.0,
    )
    .node_base(p, LayerId(1), NetClassKind::Signal);

    assert!(
        no_reference > one_sided,
        "any adjacent reference should cost less than an unreferenced inner high-speed node"
    );
    assert!(
        one_sided > dual_ground,
        "inner high-speed routing should prefer ground planes on both adjacent layers"
    );
}
