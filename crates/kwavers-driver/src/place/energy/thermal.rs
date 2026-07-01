//! Thermal-spread and airflow-blockage energy terms.
//!
//! Extracted from `compute::energy()` — lines 98–158 of the original `compute.rs`.
//! All arithmetic is bit-for-bit identical to the original; no logic was changed.

use std::collections::BTreeMap;

use super::config::{EnergyTerms, PlaceConfig};
use super::geom::{nearest_board_edge_point, segment_intersects_rect};
use crate::geom::{Nm, Point};
use crate::place::component::Component;
use crate::place::footprint::{FootprintDef, Role};

/// Accumulate thermal-spread, IC-spread, and airflow-blockage penalty terms.
///
/// # Arguments
/// * `t` — energy accumulator; `thermal`, `ic_spread`, and `airflow_blockage` are updated.
/// * `active` — centre positions of all `ActiveIc` components (pre-collected by the caller).
/// * `comps` — full component list (used for the airflow-blockage scan).
/// * `lib` — footprint library.
/// * `cfg` — placement configuration.
/// * `w` / `h` — board width/height in mm.
/// * `half_clear` — half the courtyard clearance (used for connector courtyard inflation).
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_thermal(
    t: &mut EnergyTerms,
    active: &[Point],
    comps: &[Component],
    lib: &[FootprintDef],
    cfg: &PlaceConfig,
    w: f64,
    h: f64,
    half_clear: Nm,
) {
    // Thermal: active ICs want >= thermal_spacing apart.
    let ts = cfg.thermal_spacing.to_mm();
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let d = active[i].euclid(active[j]) * 1.0e-6; // nm -> mm
            t.thermal += (ts - d).max(0.0);
        }
    }

    // IC spread: penalise the inverse of the minimum pairwise distance among same-footprint active
    // ICs. Complements the floor-based `thermal` term: once all ICs exceed `thermal_spacing` the
    // thermal term is zero, but ic_spread still provides a gradient that keeps pushing identical
    // parts apart toward a uniform distribution across the board.
    let board_diag = (w * w + h * h).sqrt();
    {
        // BTreeMap key = fp index; value = centre positions of all ActiveIc instances.
        let mut fp_positions: BTreeMap<usize, Vec<Point>> = BTreeMap::new();
        for c in comps {
            if matches!(lib[c.fp].role, Role::ActiveIc) {
                fp_positions.entry(c.fp).or_default().push(c.placement.pos);
            }
        }
        for positions in fp_positions.values() {
            if positions.len() < 2 {
                continue;
            }
            let mut min_d = f64::INFINITY;
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    // euclid returns nm-scale distance as f64; multiply by 1e-6 → mm.
                    let d = positions[i].euclid(positions[j]) * 1.0e-6;
                    min_d = min_d.min(d);
                }
            }
            // board_diag / (d + 1): O(board_diag) at d→0, → 0 as d → ∞. The +1 mm offset
            // keeps the term finite at full overlap and makes the weight dimensionless.
            t.ic_spread += board_diag / (min_d + 1.0);
        }
    }

    // Thermal airflow: keep large connector bodies out of the direct cooling corridor from the
    // nearest board edge to hot active/power devices. This is a placement objective, not a CFD
    // model; detailed thermal rise remains handled by the thermal solver.
    for (hot_idx, hot) in comps.iter().enumerate() {
        if !matches!(lib[hot.fp].role, Role::ActiveIc | Role::Power) {
            continue;
        }
        let inlet = nearest_board_edge_point(hot.placement.pos, w, h);
        for (block_idx, blocker) in comps.iter().enumerate() {
            if hot_idx == block_idx || !matches!(lib[blocker.fp].role, Role::Connector) {
                continue;
            }
            if segment_intersects_rect(
                inlet,
                hot.placement.pos,
                blocker.courtyard(lib).inflate(half_clear),
            ) {
                t.airflow_blockage += 1.0;
            }
        }
    }
}
