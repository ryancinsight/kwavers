//! Placement energy computation — the single `energy()` entry point.
//!
//! Logic is split across four focused sub-modules; this file is the orchestrator that drives them
//! in dependency order and combines the weighted total.

use crate::geom::{Nm, Point};
use crate::place::component::{Component, Rect};
use crate::place::footprint::{FootprintDef, Role};
use super::config::{CongestionField, EnergyTerms, PlaceConfig};
use super::{
    connectivity::accumulate_connectivity,
    floorplan::accumulate_floorplan,
    proximity::accumulate_proximity,
    thermal::accumulate_thermal,
};

/// Evaluate the placement energy of `comps`. `congestion`, when supplied, biases placement away
/// from regions a prior routing pass found congested (place↔route co-optimization).
pub fn energy(
    comps: &[Component],
    lib: &[FootprintDef],
    cfg: &PlaceConfig,
    congestion: Option<&CongestionField>,
) -> EnergyTerms {
    let w = cfg.board.0.to_mm();
    let h = cfg.board.1.to_mm();
    let m = cfg.margin.to_mm();

    let mut t = EnergyTerms::default();
    let half_clear = Nm(cfg.courtyard_clearance.0 / 2);

    // Fixed mechanical-feature keepouts (fiducials, mounting holes) as square keep-out regions the
    // annealer must keep component courtyards out of. Built from the same source as the router keepout
    // ([`block_mechanical`]) and emission, so the placer clears exactly the drilled/printed features.
    let mech_keepouts: Vec<Rect> = crate::geom::mechanical_features(w, h)
        .iter()
        .map(|f| {
            let r = Nm::from_mm(f.keepout_mm());
            let cx = Nm::from_mm(f.x);
            let cy = Nm::from_mm(f.y);
            Rect {
                min: Point::new(Nm(cx.0 - r.0), Nm(cy.0 - r.0)),
                max: Point::new(Nm(cx.0 + r.0), Nm(cy.0 + r.0)),
            }
        })
        .collect();

    // Overlap (all pairs) + edge keep-in + periphery + thermal (per component / per active pair).
    let mut active: Vec<Point> = Vec::new();
    for (i, c) in comps.iter().enumerate() {
        let rect = c.courtyard(lib);
        let rect_clear = rect.inflate(half_clear);
        let (x1, y1) = (rect.min.x.to_mm(), rect.min.y.to_mm());
        let (x2, y2) = (rect.max.x.to_mm(), rect.max.y.to_mm());

        // Edge keep-in: penalise courtyard crossing [m, w-m] x [m, h-m].
        t.edge += (m - x1).max(0.0)
            + (x2 - (w - m)).max(0.0)
            + (m - y1).max(0.0)
            + (y2 - (h - m)).max(0.0);

        // Periphery preference by role.
        match lib[c.fp].role {
            Role::Connector => {
                // Distance from the courtyard to the nearest board edge — want it at the rim.
                let to_edge = x1.min(w - x2).min(y1).min(h - y2).max(0.0);
                t.periphery += to_edge;
            }
            Role::ActiveIc => {
                // Distance from centre to board centre — want it in the core.
                let cx = c.placement.pos.x.to_mm();
                let cy = c.placement.pos.y.to_mm();
                let d = ((cx - w / 2.0).powi(2) + (cy - h / 2.0).powi(2)).sqrt();
                t.periphery += d;
                active.push(c.placement.pos);
            }
            _ => {}
        }

        // Mechanical-feature keepout: penalise courtyard overlap with each fiducial/mounting-hole
        // keep-out square so the part is nudged off the corners these fixed features occupy.
        for k in &mech_keepouts {
            let area = rect.overlap_area(*k);
            if area > 0.0 {
                t.mech_keepout += area * 1.0e-12; // nm² -> mm²
            }
        }

        // Overlap against later components, courtyards inflated by half the clearance each (so a
        // residual overlap means the parts are closer than `courtyard_clearance`).
        for d in comps.iter().skip(i + 1) {
            let area = rect_clear.overlap_area(d.courtyard(lib).inflate(half_clear));
            if area > 0.0 {
                t.overlap += area * 1.0e-12; // nm² -> mm²
            }
        }
    }

    let board_diag = (w * w + h * h).sqrt();

    accumulate_thermal(&mut t, &active, comps, lib, cfg, w, h, half_clear);
    accumulate_proximity(&mut t, comps, lib, cfg, half_clear);
    accumulate_connectivity(&mut t, comps, lib, cfg, half_clear, w, h, board_diag);
    let cong_weight = accumulate_floorplan(&mut t, comps, lib, cfg, congestion, half_clear, w, h);

    let wt = &cfg.weights;
    t.total = wt.overlap * t.overlap
        + wt.edge * t.edge
        + wt.periphery * t.periphery
        + wt.decoupling * t.decoupling
        + wt.termination * t.termination
        + wt.hpwl * t.hpwl
        + wt.thermal * t.thermal
        + wt.airflow_blockage * t.airflow_blockage
        + wt.utilization * t.utilization
        + wt.alignment * t.alignment
        + wt.regional * t.regional
        + wt.flow_crossing * t.flow_crossing
        + wt.channel_blockage * t.channel_blockage
        + wt.ic_spread * t.ic_spread
        + wt.isolation_drift * t.isolation_drift
        + wt.mech_keepout * t.mech_keepout
        + cong_weight * t.congestion;
    t
}
