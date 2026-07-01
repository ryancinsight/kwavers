//! Congestion-field utilities for placement feedback.
//!
//! Provides hotspot rasterisation helpers used by the board-level DFM critic
//! to build per-column penalty maps for the next placement pass.

use crate::audit::fault_report::FaultReport;
use crate::geom::{GridSpec, Nm, Point};
use crate::place::CongestionField;

/// Rasterise a set of board points into a per-column penalty field for congestion-style
/// feedback to the next placement.
///
/// One hit per point in the point's cell; shared by every feedback source (geometry
/// weaknesses, thermal hotspots, …) so they compose on one map.
#[must_use]
pub fn rasterize_hotspots(points: &[Point], spec: GridSpec, weight: f64) -> CongestionField {
    let mut per_column = vec![0.0f32; spec.nx * spec.ny];
    for &p in points {
        let (ix, iy) = spec.cell_of(p);
        per_column[iy * spec.nx + ix] += 1.0;
    }
    CongestionField {
        spec,
        per_column,
        weight,
    }
}

/// Rasterise each hotspot into a circular penalty footprint with the supplied board-space radius.
#[must_use]
pub fn rasterize_hotspots_radius(
    points: &[Point],
    spec: GridSpec,
    radius: Nm,
    weight: f64,
) -> CongestionField {
    let mut per_column = vec![0.0f32; spec.nx * spec.ny];
    let r2 = (radius.0 as i128) * (radius.0 as i128);
    for &p in points {
        let (cx, cy) = spec.cell_of(p);
        let cells = (radius.0 / spec.pitch.0).max(0) as usize + 1;
        let y0 = cy.saturating_sub(cells);
        let y1 = (cy + cells).min(spec.ny - 1);
        let x0 = cx.saturating_sub(cells);
        let x1 = (cx + cells).min(spec.nx - 1);
        for iy in y0..=y1 {
            for ix in x0..=x1 {
                let q = spec.point_of(ix, iy);
                let dx = (q.x.0 - p.x.0) as i128;
                let dy = (q.y.0 - p.y.0) as i128;
                if dx * dx + dy * dy <= r2 {
                    per_column[iy * spec.nx + ix] += 1.0;
                }
            }
        }
    }
    CongestionField {
        spec,
        per_column,
        weight,
    }
}

/// Rasterise a report's hotspots into a per-column weakness field for congestion-style
/// feedback to the next placement (the adversary's penalty map).
#[must_use]
pub fn weakness_field(report: &FaultReport, spec: GridSpec, weight: f64) -> CongestionField {
    rasterize_hotspots(&report.hotspots, spec, weight)
}
