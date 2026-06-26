//! Copper-balance (stackup warping) tests.

use super::*;

#[test]
fn copper_imbalance_is_symmetric_pair_and_counts_planes() {
    let mk = |spec: GridSpec, gl: u16, vl: u16| {
        let mut b = Board::new(spec);
        let g = b.add_net("GND", NetClassKind::Ground);
        let v = b.add_net("VPP", NetClassKind::Hv);
        let sq = vec![
            Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(2.0)),
            Point::new(Nm::from_mm(18.0), Nm::from_mm(18.0)),
            Point::new(Nm::from_mm(2.0), Nm::from_mm(18.0)),
        ];
        for (net, layer) in [(g, gl), (v, vl)] {
            b.zones.push(Zone {
                net,
                layer: LayerId(layer),
                polygon: sq.clone(),
                fill: ZoneFill::ThermalRelief,
            });
        }
        b
    };
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 4).unwrap();
    assert!(
        copper_imbalance(&mk(spec, 1, 2)) < 0.01,
        "symmetric plane placement is warp-balanced"
    );
    assert!(
        copper_imbalance(&mk(spec, 2, 3)) > 0.9,
        "planes on a non-symmetric pair are imbalanced"
    );
}

#[test]
fn copper_imbalance_flags_single_layer_routing() {
    let mut b = board();
    let n = b.add_net("N", NetClassKind::Signal);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(9.0), Nm::from_mm(1.0)),
        width: Nm::from_mm(0.3),
        layer: LayerId(0),
        net: n,
    });
    assert!(
        copper_imbalance(&b) > 0.9,
        "single-layer copper must read as imbalanced"
    );
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(1.0), Nm::from_mm(1.0)),
        end: Point::new(Nm::from_mm(9.0), Nm::from_mm(1.0)),
        width: Nm::from_mm(0.3),
        layer: LayerId(1),
        net: n,
    });
    assert!(
        copper_imbalance(&b) < 0.01,
        "mirrored copper must read as balanced"
    );
}
