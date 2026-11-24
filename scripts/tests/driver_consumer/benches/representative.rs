use kwavers_driver::{
    geom::Nm, io::write_kicad_pro, rules::DesignRules, write_kicad_pcb, Board, GridSpec, LayerId,
    NetClassKind, Point, Track,
};
use std::{hint::black_box, time::Instant};

// Repeated export of 96 routed lanes. Report a median and range instead of
// making a host-dependent time a release assertion.
fn main() {
    let mut board =
        Board::new(GridSpec::cover(Nm(30_000_000), Nm(40_000_000), Nm(250_000), 2).unwrap());
    for lane in 0..96 {
        let net = board.add_net(format!("TX_{lane}"), NetClassKind::Signal);
        let y = Nm(2_000_000 + lane * 300_000);
        board.tracks.push(Track {
            start: Point::new(Nm(2_000_000), y),
            end: Point::new(Nm(22_000_000 + lane), y),
            width: Nm(150_000),
            layer: LayerId(0),
            net,
        });
    }
    let rules = DesignRules::holohv();
    let export = || {
        black_box(write_kicad_pcb(black_box(&board), &[], &[], &rules));
        black_box(write_kicad_pro(black_box("hv_tile"), &rules));
    };
    for _ in 0..50 {
        export();
    }
    let mut times = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for _ in 0..300 {
            export();
        }
        times.push(start.elapsed().as_nanos() / 300);
    }
    times.sort_unstable();
    println!(
        "96-lane PCB+PRO export ns/op: median {}, min {}, max {}",
        times[4], times[0], times[8]
    );
}
