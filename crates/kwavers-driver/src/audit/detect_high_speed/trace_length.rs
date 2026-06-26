//! Transmission-line length (λ/10 rule) violation detector.

use std::collections::BTreeMap;

use crate::board::{Board, NetId};
use crate::geom::Point;
use crate::rules::DesignRules;

use super::super::net_util::{is_clock_like_net, is_high_speed_net, track_midpoint};

/// Detects high-speed and clock-like nets whose total routed length exceeds the **λ/10
/// transmission-line threshold** for the board's characteristic signal frequency.
///
/// # Rule
///
/// A trace whose length exceeds λ/10 = c₀ / (10·f·√εr_eff) must be treated as a
/// controlled-impedance transmission line (IPC-2141A). Without impedance matching the trace
/// acts as a signal-reflecting stub and radiates EMI proportional to the excess length. Each
/// net that crosses the threshold contributes 1 violation; its hotspot is the midpoint of the
/// last qualifying segment.
///
/// # Vacuous condition
///
/// Returns `(0, [])` when `rules.high_speed_frequency_hz <= 0` — matching the vacuous-check
/// convention used by [`crate::rules::DesignRules::ic_switching_dv_v`].
pub(crate) fn detect_transmission_line_length_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let freq = rules.high_speed_frequency_hz;
    if freq <= 0.0 {
        return (0, Vec::new());
    }
    // c₀ = 3×10⁸ m/s; 1 m = 10⁹ nm → c₀ = 3×10¹⁷ nm/s.
    const C0_NM_PER_S: f64 = 3.0e17;
    let er_eff = rules.dielectric_er.max(1.0);
    let lambda_10_nm = C0_NM_PER_S / (10.0 * freq * er_eff.sqrt());

    // Accumulate total routed length (nm) and a representative hotspot per net.
    let mut nets: BTreeMap<NetId, (f64, Point)> = BTreeMap::new();
    for t in &board.tracks {
        if !is_high_speed_net(board, t.net) && !is_clock_like_net(board, t.net) {
            continue;
        }
        let entry = nets.entry(t.net).or_insert((0.0, track_midpoint(t)));
        entry.0 += t.start.euclid(t.end);
    }

    let mut count = 0;
    let mut pts = Vec::new();
    for (total_nm, hotspot) in nets.values() {
        if *total_nm > lambda_10_nm {
            count += 1;
            pts.push(*hotspot);
        }
    }
    (count, pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{LayerId, NetClassKind, Track};
    use crate::geom::{GridSpec, Nm};

    fn make_board_with_track(
        net_name: &str,
        class: NetClassKind,
        track_length_mm: f64,
    ) -> Board {
        let spec = GridSpec::cover(
            Nm::from_mm(100.0),
            Nm::from_mm(100.0),
            Nm::from_mm(0.5),
            2,
        )
        .unwrap();
        let mut b = Board::new(spec);
        let net = b.add_net(net_name, class);
        let len = Nm::from_mm(track_length_mm);
        b.tracks.push(Track {
            start: crate::geom::Point::new(Nm(0), Nm(0)),
            end: crate::geom::Point::new(len, Nm(0)),
            width: Nm::from_mm(0.2),
            net,
            layer: LayerId(0),
        });
        b
    }

    /// At 100 MHz with εr = 4.5: λ/10 = 3×10¹⁷ / (10 · 10⁸ · √4.5) ≈ 141 mm.
    /// A CLK trace of 200 mm must be flagged; 50 mm must pass.
    #[test]
    fn flags_clk_trace_exceeding_lambda_10() {
        let long = make_board_with_track("CLK", NetClassKind::Signal, 200.0);
        let short = make_board_with_track("CLK", NetClassKind::Signal, 50.0);
        let mut rules = DesignRules::holohv();
        rules.high_speed_frequency_hz = 1.0e8;
        rules.dielectric_er = 4.5;
        let (n_long, _) = detect_transmission_line_length_violations(&long, &rules);
        let (n_short, _) = detect_transmission_line_length_violations(&short, &rules);
        assert_eq!(n_long, 1, "200 mm CLK trace at 100 MHz must be flagged");
        assert_eq!(n_short, 0, "50 mm CLK trace at 100 MHz must pass");
    }

    /// Nets that are neither high-speed nor clock-like must not be flagged
    /// even when their length exceeds λ/10.
    #[test]
    fn ignores_non_high_speed_nets() {
        let b = make_board_with_track("VCC", NetClassKind::Signal, 500.0);
        let rules = DesignRules::holohv();
        let (n, _) = detect_transmission_line_length_violations(&b, &rules);
        assert_eq!(n, 0, "power net VCC must not be flagged by λ/10 check");
    }

    /// Zero frequency makes the check vacuous.
    #[test]
    fn vacuous_when_frequency_is_zero() {
        let b = make_board_with_track("CLK", NetClassKind::Signal, 1000.0);
        let mut rules = DesignRules::holohv();
        rules.high_speed_frequency_hz = 0.0;
        let (n, _) = detect_transmission_line_length_violations(&b, &rules);
        assert_eq!(n, 0, "zero frequency makes check vacuous");
    }
}
