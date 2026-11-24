use kwavers_driver::{
    geom::Nm, Board, GridSpec, LayerId, NetClassKind, Point, Track, Via, ViaKind,
};
use tracing_subscriber::fmt::format::FmtSpan;

pub fn trace(name: &'static str) -> tracing::span::EnteredSpan {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .without_time()
        .with_ansi(false)
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .try_init();
    tracing::info_span!("consumer", case = name).entered()
}

// Straight horizontal runs make the length oracle exact in integer nanometres.
// Interior vertices change layers via a single microvia, as in the published fixture.
pub fn pair(p_vertices: &[i64], n_vertices: &[i64]) -> Board {
    let spec = GridSpec::cover(Nm(20_000_000), Nm(20_000_000), Nm(500_000), 2).unwrap();
    let mut board = Board::new(spec);
    for (name, y, vertices) in [
        ("DATA_P", 4_000_000, p_vertices),
        ("DATA_N", 5_000_000, n_vertices),
    ] {
        let net = board.add_net(name, NetClassKind::Signal);
        for (layer, segment) in vertices.windows(2).enumerate() {
            let start = Point::new(Nm(segment[0]), Nm(y));
            let end = Point::new(Nm(segment[1]), Nm(y));
            board.tracks.push(Track {
                start,
                end,
                width: Nm(150_000),
                layer: LayerId(layer as u16),
                net,
            });
            if layer > 0 {
                board.vias.push(Via {
                    pos: start,
                    drill: Nm(200_000),
                    diameter: Nm(460_000),
                    net,
                    from: LayerId(0),
                    to: LayerId(1),
                    kind: ViaKind::Micro,
                    filled: false,
                });
            }
        }
    }
    board
}

// Positive expressions here use at most 12 rounded operations, including literal conversion.
// 16 eps exceeds gamma_12 = 12u/(1-12u), u=eps/2, including reference rounding.
pub fn near(value: f64, expected: f64) {
    assert!(
        (value - expected).abs() <= expected.abs() * 16.0 * f64::EPSILON,
        "value={value:.17e}, expected={expected:.17e}"
    );
}
