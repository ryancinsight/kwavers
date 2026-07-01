//! Reference-plane / split-domain high-speed audit tests.

use super::super::board;
use crate::audit::audit;
use crate::board::{Board, LayerId, NetClassKind, Pad, Track, Zone, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::rules::DesignRules;

#[test]
fn detects_signal_track_intruding_on_reference_plane() {
    let mut b = board();
    let tx = b.add_net("TX_PLANE_CUT", NetClassKind::Signal);
    let ctrl = b.add_net("CTRL_OTHER_LAYER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: tx,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(12.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(12.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: ctrl,
    });
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(14.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(14.0)),
        width: Nm::from_mm(0.25),
        layer: LayerId(1),
        net: gnd,
    });

    let r = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        r.reference_plane_intrusion_violations, 1,
        "only the non-plane signal track routed through the reference-plane zone is flagged"
    );
}

#[test]
fn detects_signal_crossing_opposite_split_ground_domain() {
    let mut b = board();
    let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
    let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
    let agnd = b.add_net("AGND", NetClassKind::Ground);
    let dgnd = b.add_net("DGND", NetClassKind::Ground);
    let zone = |net, x0: f64, x1: f64| Zone {
        net,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(x0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(x0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };
    b.zones.push(zone(agnd, 2.0, 10.0));
    b.zones.push(zone(dgnd, 10.0, 18.0));
    let route = |board: &mut Board, net, x0: f64, x1: f64, y: f64| {
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(1),
            net,
        });
    };

    let mut clean = b.clone();
    route(&mut clean, analog, 3.0, 9.0, 6.0);
    route(&mut clean, digital, 11.0, 17.0, 8.0);
    let same_domain = audit(&clean, &[], &[], &DesignRules::holohv());
    assert_eq!(
        same_domain.split_domain_reference_violations, 0,
        "analog over AGND and digital over DGND stay inside their split-ground domains"
    );

    let mut crossed = b;
    route(&mut crossed, digital, 3.0, 9.0, 6.0);
    route(&mut crossed, analog, 11.0, 17.0, 8.0);
    let opposite_domain = audit(&crossed, &[], &[], &DesignRules::holohv());
    assert_eq!(
        opposite_domain.split_domain_reference_violations, 2,
        "digital over AGND and analog over DGND both violate split-domain reference routing"
    );
}

#[test]
fn detects_mixed_domain_shared_ground_return_overlap() {
    let mut b = board();
    let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
    let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let route = |board: &mut Board, net, y: f64| {
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(3.0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(17.0), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    };

    let mut separated = b.clone();
    route(&mut separated, analog, 6.0);
    route(&mut separated, digital, 8.0);
    let clean = audit(&separated, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.mixed_domain_shared_reference_violations, 0,
        "analog and digital returns sharing GND but separated by more than the sensitive keepout are clean"
    );

    route(&mut b, analog, 6.0);
    route(&mut b, digital, 6.8);
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.mixed_domain_shared_reference_violations, 1,
        "analog and digital tracks closer than the sensitive keepout over one GND plane overlap return-current corridors"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "mixed-domain return-current overlap must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_virtual_split_domain_crossing() {
    let mut b = board();
    let analog = b.add_net("ANALOG_SIG", NetClassKind::Signal);
    let digital = b.add_net("DIGITAL_SIG", NetClassKind::Signal);
    for (net, x) in [(analog, 4.0), (digital, 16.0)] {
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(4.0)),
            layers: vec![LayerId(0)],
            net: Some(net),
        });
        b.add_pad(Pad {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(16.0)),
            layers: vec![LayerId(0)],
            net: Some(net),
        });
    }
    let route = |board: &mut Board, net, x0: f64, x1: f64, y: f64| {
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(x0), Nm::from_mm(y)),
            end: Point::new(Nm::from_mm(x1), Nm::from_mm(y)),
            width: Nm::from_mm(0.15),
            layer: LayerId(0),
            net,
        });
    };

    let mut clean_board = b.clone();
    route(&mut clean_board, analog, 3.0, 8.0, 6.0);
    route(&mut clean_board, digital, 12.0, 17.0, 14.0);
    let clean = audit(&clean_board, &[], &[], &DesignRules::holohv());
    assert_eq!(
        clean.virtual_split_crossing_violations, 0,
        "analog and digital tracks staying on their centroid-derived virtual sides are clean"
    );

    route(&mut b, analog, 3.0, 12.0, 6.0);
    route(&mut b, digital, 12.0, 17.0, 14.0);
    let dirty = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dirty.virtual_split_crossing_violations, 1,
        "the analog track crossing the inferred x=10 mm virtual split line is flagged"
    );
    assert!(
        !dirty.hard_drc_clean(),
        "virtual split crossing must reject optimizer clean-board selection"
    );
}

#[test]
fn detects_fragmented_same_net_ground_plane() {
    let mut b = board();
    let gnd = b.add_net("GND", NetClassKind::Ground);
    let island = |x0: f64, x1: f64| Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(x0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(x1), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(x0), Nm::from_mm(18.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };

    b.zones.push(island(2.0, 18.0));
    let continuous = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        continuous.ground_plane_fragmentation_violations, 0,
        "one ground pour on a layer is a continuous reference plane"
    );

    b.zones.push(island(20.0, 24.0));
    let fragmented = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        fragmented.ground_plane_fragmentation_violations, 1,
        "two same-net ground pour islands on one layer fragment the reference plane"
    );

    b.zones[1].fill = ZoneFill::Solid;
    let teardrop_like = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        teardrop_like.ground_plane_fragmentation_violations, 0,
        "solid same-net reinforcement zones are not counted as plane-pour fragmentation"
    );
}

#[test]
fn detects_high_speed_track_without_adjacent_reference_plane() {
    let mut b = board();
    let tx = b.add_net("TX_REF", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(0),
        net: tx,
    });

    let missing = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        missing.reference_plane_absence_violations, 1,
        "a high-speed segment with no adjacent reference zone is flagged"
    );

    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(0),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let same_layer = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        same_layer.reference_plane_absence_violations, 1,
        "same-layer copper is not an adjacent reference plane"
    );

    b.zones.clear();
    b.zones.push(Zone {
        net: gnd,
        layer: LayerId(1),
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(10.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    });
    let partial = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        partial.reference_plane_absence_violations, 1,
        "adjacent reference coverage must cover the segment start, middle, and end"
    );

    b.zones[0].polygon = vec![
        Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
        Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
        Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
        Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
    ];
    let covered = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        covered.reference_plane_absence_violations, 0,
        "a full adjacent ground zone supplies the high-speed reference plane"
    );
}

#[test]
fn detects_inner_high_speed_track_without_dual_ground_reference() {
    let spec = GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    let mut b = Board::new(spec);
    let tx = b.add_net("TX_INNER", NetClassKind::Signal);
    let gnd = b.add_net("GND", NetClassKind::Ground);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(4.0), Nm::from_mm(10.0)),
        end: Point::new(Nm::from_mm(16.0), Nm::from_mm(10.0)),
        width: Nm::from_mm(0.15),
        layer: LayerId(1),
        net: tx,
    });
    let reference = |layer| Zone {
        net: gnd,
        layer,
        polygon: vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(8.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(12.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(12.0)),
        ],
        fill: ZoneFill::ThermalRelief,
    };
    b.zones.push(reference(LayerId(0)));
    let one_sided = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        one_sided.reference_plane_absence_violations, 0,
        "one adjacent ground zone satisfies the generic adjacent-reference rule"
    );
    assert_eq!(
        one_sided.inner_layer_dual_ground_reference_violations, 1,
        "an inner high-speed layer requires ground reference zones on both adjacent layers"
    );

    b.zones.push(reference(LayerId(2)));
    let dual = audit(&b, &[], &[], &DesignRules::holohv());
    assert_eq!(
        dual.inner_layer_dual_ground_reference_violations, 0,
        "ground zones on both adjacent layers clear the inner-layer high-speed reference rule"
    );
}
