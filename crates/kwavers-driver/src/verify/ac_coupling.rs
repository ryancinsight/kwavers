//! AC Coupling — high-frequency coplanar edge coupling check.
use crate::board::{Board, NetClassKind, NetId};
use crate::geom::{Nm, Point};
use std::collections::HashMap;

/// Parasitic AC coupling violation: low impedance at edge rate between switching line and GND.
#[derive(Debug, Clone, PartialEq)]
pub struct AcCouplingViolation {
    /// High-speed/HV switching net name.
    pub switching_net: String,
    /// Reference ground net name.
    pub ground_net: String,
    /// Estimated parasitic capacitance in pF.
    pub capacitance_pf: f64,
    /// Coupling impedance at switching edge rate.
    pub impedance_ohm: f64,
}

/// Results of parasitic AC coupling check.
#[derive(Debug, Clone, Default)]
pub struct AcCouplingReport {
    /// List of violations.
    pub violations: Vec<AcCouplingViolation>,
    /// Hotspot midpoints of violating segments.
    pub hotspots: Vec<Point>,
    /// True if all switching nets have coupling impedance >= 1000 Ohms.
    pub pass: bool,
}

/// Estimates parasitic capacitance and impedance from high dV/dt switching tracks
/// to adjacent coplanar ground tracks on the same layer.
#[must_use]
pub fn parasitic_ac_coupling_check(board: &Board) -> AcCouplingReport {
    let is_switching = |net: NetId| {
        let name = &board.nets[net.0 as usize].name;
        board.class_of(net) == NetClassKind::Hv
            || name.starts_with("TRIG")
            || name.starts_with("OUT")
            || name.starts_with("TX")
    };

    let is_ground = |net: NetId| board.class_of(net) == NetClassKind::Ground;

    let mut net_caps: HashMap<NetId, f64> = HashMap::new();
    let mut net_hotspots: HashMap<NetId, Vec<Point>> = HashMap::new();
    let tr = &board.tracks;

    for i in 0..tr.len() {
        let ti = &tr[i];
        let w_i = ti.width.0 as f64 * 1.0e-6;
        let is_i_sw = is_switching(ti.net);
        let is_i_gnd = is_ground(ti.net);

        if !is_i_sw && !is_i_gnd {
            continue;
        }

        for tj in tr.iter().skip(i + 1) {
            if ti.layer != tj.layer {
                continue;
            }
            let is_j_sw = is_switching(tj.net);
            let is_j_gnd = is_ground(tj.net);

            let (sw_net, _gnd_net) = if is_i_sw && is_j_gnd {
                (ti.net, tj.net)
            } else if is_j_sw && is_i_gnd {
                (tj.net, ti.net)
            } else {
                continue;
            };

            let (a, b) = (ti.start, ti.end);
            let (c_pt, d_pt) = (tj.start, tj.end);
            let horiz_i = a.y.0 == b.y.0;
            let vert_i = a.x.0 == b.x.0;
            let horiz_j = c_pt.y.0 == d_pt.y.0;
            let vert_j = c_pt.x.0 == d_pt.x.0;

            let mut overlap_nm = 0.0;
            let mut spacing_nm = 0.0;
            let mut midpoint = Point::new(Nm(0), Nm(0));

            if horiz_i && horiz_j {
                let overlap_min = a.x.0.min(b.x.0).max(c_pt.x.0.min(d_pt.x.0));
                let overlap_max = a.x.0.max(b.x.0).min(c_pt.x.0.max(d_pt.x.0));
                let overlap = (overlap_max - overlap_min) as f64;
                if overlap > 0.0 {
                    overlap_nm = overlap;
                    spacing_nm = ((a.y.0 - c_pt.y.0).abs() as f64)
                        - (ti.width.0 as f64 + tj.width.0 as f64) / 2.0;
                    let mid_x = (overlap_min + overlap_max) / 2;
                    let mid_y = (a.y.0 + c_pt.y.0) / 2;
                    midpoint = Point::new(Nm(mid_x), Nm(mid_y));
                }
            } else if vert_i && vert_j {
                let overlap_min = a.y.0.min(b.y.0).max(c_pt.y.0.min(d_pt.y.0));
                let overlap_max = a.y.0.max(b.y.0).min(c_pt.y.0.max(d_pt.y.0));
                let overlap = (overlap_max - overlap_min) as f64;
                if overlap > 0.0 {
                    overlap_nm = overlap;
                    spacing_nm = ((a.x.0 - c_pt.x.0).abs() as f64)
                        - (ti.width.0 as f64 + tj.width.0 as f64) / 2.0;
                    let mid_x = (a.x.0 + c_pt.x.0) / 2;
                    let mid_y = (overlap_min + overlap_max) / 2;
                    midpoint = Point::new(Nm(mid_x), Nm(mid_y));
                }
            }

            if overlap_nm > 0.0 && spacing_nm > 0.0 {
                let spacing = spacing_nm * 1.0e-6;
                if spacing > 0.0 && spacing <= 2.0 {
                    let len = overlap_nm * 1.0e-6;
                    let c_seg = len * 0.008 * (1.0 + 2.0 * w_i / spacing).ln();
                    *net_caps.entry(sw_net).or_default() += c_seg;
                    net_hotspots.entry(sw_net).or_default().push(midpoint);
                }
            }
        }
    }

    let mut violations = Vec::new();
    let mut hotspots = Vec::new();
    for (sw_net, c_total) in net_caps {
        if c_total > 0.0 {
            let z = 13642.0 / c_total;
            if z < 1000.0 {
                let sw_name = board.nets[sw_net.0 as usize].name.clone();
                violations.push(AcCouplingViolation {
                    switching_net: sw_name,
                    ground_net: "GND".to_string(),
                    capacitance_pf: c_total,
                    impedance_ohm: z,
                });
                if let Some(pts) = net_hotspots.get(&sw_net) {
                    hotspots.extend(pts.clone());
                }
            }
        }
    }

    violations.sort_by(|a, b| a.switching_net.cmp(&b.switching_net));
    hotspots.sort_by(|a, b| a.x.cmp(&b.x).then_with(|| a.y.cmp(&b.y)));
    hotspots.dedup();
    let pass = violations.is_empty();
    AcCouplingReport {
        violations,
        hotspots,
        pass,
    }
}
