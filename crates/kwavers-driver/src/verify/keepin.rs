//! Keep-in — component-to-board-edge fit.
use crate::board::Board;
use crate::geom::Nm;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;

/// Board keep-in findings: components whose placed courtyard crosses the board-edge keep-out.
#[derive(Debug, Clone, Default)]
pub struct KeepinReport {
    /// `(refdes, worst overhang past the keep-in boundary in mm)` for each offending component.
    pub edge_violations: Vec<(String, f64)>,
    /// `(feature, shortfall mm)` for routed copper whose outer edge enters the board-edge keep-out.
    pub copper_edge_violations: Vec<(String, f64)>,
    /// Whether every component and routed copper feature sits inside the board minus the edge margin.
    pub pass: bool,
}

/// Check that every component's placed courtyard and every routed copper feature lies inside the
/// board outline minus `margin` — the mechanical/electrical keep-in. Copper or a package body inside
/// the edge keep-out risks exposure at the routed/V-scored outline (a fab reject) or a mechanical
/// clash with the enclosure. The board extent is taken from the routing grid (`origin` to the last
/// cell centre).
#[must_use]
pub fn keepin(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    margin: Nm,
) -> KeepinReport {
    let spec = board.spec;
    let min_x = spec.origin.x.0 + margin.0;
    let min_y = spec.origin.y.0 + margin.0;
    let max_x = spec.origin.x.0 + (spec.nx as i64 - 1) * spec.pitch.0 - margin.0;
    let max_y = spec.origin.y.0 + (spec.ny as i64 - 1) * spec.pitch.0 - margin.0;
    let mut edge_violations = Vec::new();
    let mut copper_edge_violations = Vec::new();
    for c in comps {
        let cy = c.courtyard(lib);
        // Worst overhang past any of the four keep-in boundaries (positive = outside).
        let over = [
            min_x - cy.min.x.0,
            min_y - cy.min.y.0,
            cy.max.x.0 - max_x,
            cy.max.y.0 - max_y,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        if over > 0 {
            edge_violations.push((c.refdes.clone(), over as f64 * 1.0e-6));
        }
    }
    for (i, t) in board.tracks.iter().enumerate() {
        let half = t.width.0 / 2;
        let center_clearance = [
            t.start.x.0.min(t.end.x.0) - spec.origin.x.0,
            t.start.y.0.min(t.end.y.0) - spec.origin.y.0,
            spec.origin.x.0 + (spec.nx as i64 - 1) * spec.pitch.0 - t.start.x.0.max(t.end.x.0),
            spec.origin.y.0 + (spec.ny as i64 - 1) * spec.pitch.0 - t.start.y.0.max(t.end.y.0),
        ]
        .into_iter()
        .min()
        .unwrap_or(i64::MAX);
        let copper_clearance = center_clearance - half;
        let shortfall = margin.0 - copper_clearance;
        if shortfall > 0 {
            copper_edge_violations.push((format!("track[{i}]"), shortfall as f64 * 1.0e-6));
        }
    }
    for (i, v) in board.vias.iter().enumerate() {
        let radius = v.diameter.0 / 2;
        let center_clearance = [
            v.pos.x.0 - spec.origin.x.0,
            v.pos.y.0 - spec.origin.y.0,
            spec.origin.x.0 + (spec.nx as i64 - 1) * spec.pitch.0 - v.pos.x.0,
            spec.origin.y.0 + (spec.ny as i64 - 1) * spec.pitch.0 - v.pos.y.0,
        ]
        .into_iter()
        .min()
        .unwrap_or(i64::MAX);
        let copper_clearance = center_clearance - radius;
        let shortfall = margin.0 - copper_clearance;
        if shortfall > 0 {
            copper_edge_violations.push((format!("via[{i}]"), shortfall as f64 * 1.0e-6));
        }
    }
    edge_violations.sort_by(|a, b| a.0.cmp(&b.0));
    copper_edge_violations.sort_by(|a, b| a.0.cmp(&b.0));
    KeepinReport {
        pass: edge_violations.is_empty() && copper_edge_violations.is_empty(),
        edge_violations,
        copper_edge_violations,
    }
}
