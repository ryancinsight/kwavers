//! Ground-via stitching density and signal-via return-count violation detectors.
//!
//! Both rules are drawn from TI Application Note SLYP173, Section 5-12/5-17:
//!
//! * **Stitching density**: "Ground vias should be spaced no more than 0.25 cm (0.1″) apart
//!   for the entire trace length" — every point on a high-speed track must be within
//!   [`crate::rules::DesignRules::max_ground_via_stitching_spacing`] of a ground via.
//!
//! * **Return-via count**: "Using 4 ground vias next to a signal via shows very good results"
//!   — each high-speed layer-transition via must have at least
//!   [`crate::rules::DesignRules::min_signal_via_return_count`] ground transition vias within
//!   the [`crate::rules::DesignRules::high_speed_transition_ground_via_distance`] search
//!   radius. The existing [`crate::audit::detect_high_speed::detect_high_speed_transition_ground_via_violations`]
//!   check requires ≥ 1; this check enforces a configurable minimum that can be raised to the
//!   TI-recommended 4.

use crate::board::{Board, NetClassKind};
use crate::geom::{Nm, Point};
use crate::rules::DesignRules;

use super::super::net_util::{is_clock_like_net, is_high_speed_net};

/// Flags high-speed/clock track segments where no ground via is within
/// `rules.max_ground_via_stitching_spacing` of every sampled point along the segment.
///
/// The segment is sampled at intervals of `spacing / 2`, so a long trace is covered by
/// several independent sample points — any sample that is farther than the spacing from the
/// nearest ground via causes its segment to be flagged once, at the failing sample point.
///
/// # Vacuous condition
///
/// Returns `(0, [])` when `rules.max_ground_via_stitching_spacing` is zero or negative.
pub(crate) fn detect_ground_via_stitching_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    let spacing_nm = rules.max_ground_via_stitching_spacing.0 as f64;
    if spacing_nm <= 0.0 {
        return (0, Vec::new());
    }

    let ground_via_pts: Vec<Point> = board
        .vias
        .iter()
        .filter(|v| board.class_of(v.net) == NetClassKind::Ground)
        .map(|v| v.pos)
        .collect();

    let mut count = 0;
    let mut pts = Vec::new();

    for t in &board.tracks {
        if !is_high_speed_net(board, t.net) && !is_clock_like_net(board, t.net) {
            continue;
        }
        let seg_len = t.start.euclid(t.end);
        // Sample at intervals of spacing/2 so a single via can cover the full half-wavelength
        // window on both sides. Minimum 1 sample at the midpoint for zero-length segments.
        let n_samples = ((seg_len / (spacing_nm * 0.5)).ceil() as usize).max(1);
        let mut failed_at: Option<Point> = None;
        for i in 0..=n_samples {
            let frac = i as f64 / n_samples as f64;
            let sample = Point::new(
                Nm((t.start.x.0 as f64 + (t.end.x.0 - t.start.x.0) as f64 * frac) as i64),
                Nm((t.start.y.0 as f64 + (t.end.y.0 - t.start.y.0) as f64 * frac) as i64),
            );
            let nearest = ground_via_pts
                .iter()
                .fold(f64::INFINITY, |acc, &g| acc.min(sample.euclid(g)));
            if nearest > spacing_nm {
                failed_at = Some(sample);
                break;
            }
        }
        if let Some(vp) = failed_at {
            count += 1;
            pts.push(vp);
        }
    }
    (count, pts)
}

/// Flags high-speed layer-transition vias that have fewer than
/// `rules.min_signal_via_return_count` ground transition vias within
/// `rules.high_speed_transition_ground_via_distance`.
///
/// TI SLYP173 §5-15/5-17: "4 ground vias next to a signal via shows very good results."
/// The existing transition-ground-via check requires ≥ 1 return via (any-or-nothing);
/// this check enforces the configurable minimum count, allowing the rule to be tightened
/// toward the TI-recommended 4 for critical nets.
///
/// # Vacuous condition
///
/// Returns `(0, [])` when `rules.min_signal_via_return_count == 0`.
pub(crate) fn detect_high_speed_via_return_count_violations(
    board: &Board,
    rules: &DesignRules,
) -> (usize, Vec<Point>) {
    if rules.min_signal_via_return_count == 0 {
        return (0, Vec::new());
    }
    let max_dist = rules.high_speed_transition_ground_via_distance.0 as f64;
    let min_count = rules.min_signal_via_return_count;

    let ground_transition_vias: Vec<Point> = board
        .vias
        .iter()
        .filter(|v| v.from != v.to && board.class_of(v.net) == NetClassKind::Ground)
        .map(|v| v.pos)
        .collect();

    let mut count = 0;
    let mut pts = Vec::new();

    for via in &board.vias {
        if via.from == via.to
            || (!is_high_speed_net(board, via.net) && !is_clock_like_net(board, via.net))
        {
            continue;
        }
        let return_count = ground_transition_vias
            .iter()
            .filter(|&&g| via.pos.euclid(g) <= max_dist)
            .count();
        if return_count < min_count {
            count += 1;
            pts.push(via.pos);
        }
    }
    (count, pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{LayerId, NetClassKind, Track, Via, ViaKind};
    use crate::geom::{GridSpec, Nm};

    fn make_board() -> Board {
        let spec =
            GridSpec::cover(Nm::from_mm(50.0), Nm::from_mm(50.0), Nm::from_mm(0.5), 2).unwrap();
        Board::new(spec)
    }

    // ---- stitching density tests ----------------------------------------

    /// A 1 mm CLK track with a ground via at (0,0) and the track running from (0,0) to (1,0)
    /// in mm. spacing = 2.5 mm → the midpoint (0.5 mm) is within 2.5 mm → passes.
    #[test]
    fn passes_clk_track_with_nearby_ground_via() {
        let mut b = make_board();
        let hs_net = b.add_net("CLK", NetClassKind::Signal);
        let gnd_net = b.add_net("GND", NetClassKind::Ground);
        b.tracks.push(Track {
            start: Point::new(Nm(0), Nm(0)),
            end: Point::new(Nm::from_mm(1.0), Nm(0)),
            width: Nm::from_mm(0.2),
            net: hs_net,
            layer: LayerId(0),
        });
        b.vias.push(Via {
            pos: Point::new(Nm(0), Nm(0)),
            net: gnd_net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Through,
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            filled: false,
        });
        let mut rules = DesignRules::holohv();
        rules.max_ground_via_stitching_spacing = Nm::from_mm(2.5);
        let (n, _) = detect_ground_via_stitching_violations(&b, &rules);
        assert_eq!(n, 0, "track within 2.5 mm of ground via must pass");
    }

    /// A 10 mm CLK track with no ground vias → every sample point fails → 1 violation.
    #[test]
    fn flags_clk_track_with_no_nearby_ground_via() {
        let mut b = make_board();
        let hs_net = b.add_net("CLK", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm(0), Nm(0)),
            end: Point::new(Nm::from_mm(10.0), Nm(0)),
            width: Nm::from_mm(0.2),
            net: hs_net,
            layer: LayerId(0),
        });
        let mut rules = DesignRules::holohv();
        rules.max_ground_via_stitching_spacing = Nm::from_mm(2.5);
        let (n, _) = detect_ground_via_stitching_violations(&b, &rules);
        assert_eq!(n, 1, "CLK track with no ground via must be flagged");
    }

    /// Non-high-speed nets are ignored even when far from all ground vias.
    #[test]
    fn ignores_non_high_speed_tracks() {
        let mut b = make_board();
        let sig_net = b.add_net("UART_TX", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm(0), Nm(0)),
            end: Point::new(Nm::from_mm(10.0), Nm(0)),
            width: Nm::from_mm(0.2),
            net: sig_net,
            layer: LayerId(0),
        });
        let mut rules = DesignRules::holohv();
        rules.max_ground_via_stitching_spacing = Nm::from_mm(2.5);
        let (n, _) = detect_ground_via_stitching_violations(&b, &rules);
        assert_eq!(n, 0, "non-high-speed net must not be flagged");
    }

    /// Zero spacing makes the check vacuous.
    #[test]
    fn vacuous_when_spacing_is_zero() {
        let mut b = make_board();
        let hs_net = b.add_net("CLK", NetClassKind::Signal);
        b.tracks.push(Track {
            start: Point::new(Nm(0), Nm(0)),
            end: Point::new(Nm::from_mm(20.0), Nm(0)),
            width: Nm::from_mm(0.2),
            net: hs_net,
            layer: LayerId(0),
        });
        let mut rules = DesignRules::holohv();
        rules.max_ground_via_stitching_spacing = Nm(0);
        let (n, _) = detect_ground_via_stitching_violations(&b, &rules);
        assert_eq!(n, 0, "zero spacing makes check vacuous");
    }

    // ---- return-via count tests ------------------------------------------

    /// A high-speed layer-transition via with 2 ground return vias nearby passes min_count=2.
    #[test]
    fn passes_signal_via_with_sufficient_return_vias() {
        let mut b = make_board();
        let hs_net = b.add_net("CLK", NetClassKind::Signal);
        let gnd_net = b.add_net("GND", NetClassKind::Ground);
        // Signal via at (10,10)
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            net: hs_net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Through,
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            filled: false,
        });
        // Two ground transition vias nearby
        for dx in [0.5_f64, -0.5_f64] {
            b.vias.push(Via {
                pos: Point::new(Nm::from_mm(10.0 + dx), Nm::from_mm(10.0)),
                net: gnd_net,
                from: LayerId(0),
                to: LayerId(1),
                kind: ViaKind::Through,
                drill: Nm::from_mm(0.3),
                diameter: Nm::from_mm(0.5),
                filled: false,
            });
        }
        let mut rules = DesignRules::holohv();
        rules.min_signal_via_return_count = 2;
        let (n, _) = detect_high_speed_via_return_count_violations(&b, &rules);
        assert_eq!(n, 0, "2 return vias for min_count=2 must pass");
    }

    /// A high-speed layer-transition via with only 1 ground return via fails min_count=2.
    #[test]
    fn flags_signal_via_with_insufficient_return_vias() {
        let mut b = make_board();
        let hs_net = b.add_net("CLK", NetClassKind::Signal);
        let gnd_net = b.add_net("GND", NetClassKind::Ground);
        // Signal via at origin
        b.vias.push(Via {
            pos: Point::new(Nm(0), Nm(0)),
            net: hs_net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Through,
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            filled: false,
        });
        // Only ONE ground via nearby
        b.vias.push(Via {
            pos: Point::new(Nm::from_mm(0.5), Nm(0)),
            net: gnd_net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Through,
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            filled: false,
        });
        let mut rules = DesignRules::holohv();
        rules.min_signal_via_return_count = 2;
        let (n, _) = detect_high_speed_via_return_count_violations(&b, &rules);
        assert_eq!(n, 1, "only 1 return via for min_count=2 must be flagged");
    }

    /// min_signal_via_return_count == 0 makes the check vacuous.
    #[test]
    fn vacuous_when_min_count_is_zero() {
        let mut b = make_board();
        let hs_net = b.add_net("CLK", NetClassKind::Signal);
        b.vias.push(Via {
            pos: Point::new(Nm(0), Nm(0)),
            net: hs_net,
            from: LayerId(0),
            to: LayerId(1),
            kind: ViaKind::Through,
            drill: Nm::from_mm(0.3),
            diameter: Nm::from_mm(0.5),
            filled: false,
        });
        let mut rules = DesignRules::holohv();
        rules.min_signal_via_return_count = 0;
        let (n, _) = detect_high_speed_via_return_count_violations(&b, &rules);
        assert_eq!(n, 0, "min_count=0 makes check vacuous");
    }
}
