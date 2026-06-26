//! `RoutingCost` trait bridge for `PhysicsCost`.
//!
//! (Plain backticks rather than `[`X`](path)` intra-doc links throughout this file: rustdoc's
//! `redundant_explicit_links` lint rejects clickable `[`X`](...X...)` forms where the
//! resolution is unambiguous from the link text alone.)
//!
//! This file owns `impl RoutingCost for PhysicsCost`. Reading the file in isolation shows the
//! full per-class penalty contract: which penalty fields are read from `PhysicsCost`'s
//! precomputed arrays, how they're weighted, and the via-cost multiplier that distinguishes
//! high-speed classes from plane-like ones.
//!
//! Keeping the trait impl in its own file is intentional — the dependency-inversion bridge
//! between the `RoutingCost` trait seam and the `PhysicsCost` concrete model becomes a
//! single-file unit a reviewer can audit independently of the `PhysicsCost::new` builder and
//! the `RoutingCost` trait definition.
//!
//! Per-class penalty contract (read this file's `node_base` matrix):
//! * `Hv` and `Signal` pay **all** penalty terms: creepage proximity (HV vs LV; LV vs HV),
//!   reference-plane presence + margin, inner dual-ground, power-reference, high-speed edge,
//!   high-speed spacing (same + adjacent layer), top-side outer, layer affinity.
//! * `Power` and `Ground` pay **none**: their `node_base` collapses to
//!   `1.0 + self.creepage_weight * hazard` where `hazard` is `hv_field`/`lv_field` for the
//!   class (HV maps to `lv_field`, everything else to `hv_field`) — but plane-like classes are
//!   routed on dedicated planes so the per-cell physics is immaterial.
//! * `via_cost` is class-split: Hv/Signal pay `via_cost * HIGH_SPEED_VIA_MULTIPLIER` (2×);
//!   Power/Ground pay the base `via_cost`.

use crate::board::{LayerId, NetClassKind};
use crate::geom::Point;

use super::physics::{
    PhysicsCost, HIGH_SPEED_EDGE_PENALTY, HIGH_SPEED_SPACING_PENALTY, HIGH_SPEED_VIA_MULTIPLIER,
    INNER_DUAL_GROUND_PENALTY, POWER_REFERENCE_PENALTY, REFERENCE_MARGIN_PENALTY,
    REFERENCE_PLANE_PENALTY,
};
use super::routing_cost::RoutingCost;

impl RoutingCost for PhysicsCost {
    fn node_base(&self, p: Point, layer: LayerId, class: NetClassKind) -> f64 {
        let (ix, iy) = self.spec.cell_of(p);
        let c = iy * self.spec.nx + ix;
        // Hoist layer-dependent offset used by every per-layer penalty below.
        let layer_idx = layer.0 as usize;
        let lc = layer_idx * self.spec.nx * self.spec.ny + c;
        let hazard = match class {
            // HV avoids low-voltage features.
            NetClassKind::Hv => self.lv_field[c] as f64,
            // LV avoids high-voltage features.
            _ => self.hv_field[c] as f64,
        };
        let reference_penalty = match class {
            NetClassKind::Hv | NetClassKind::Signal => {
                self.reference_field[lc] as f64 * REFERENCE_PLANE_PENALTY
            }
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        };
        let reference_margin_penalty = match class {
            NetClassKind::Hv | NetClassKind::Signal => {
                let budget = match class {
                    NetClassKind::Hv => self.hv_reference_margin_budget,
                    NetClassKind::Signal => self.signal_reference_margin_budget,
                    NetClassKind::Power | NetClassKind::Ground => 0.0,
                };
                let margin = self.reference_margin_nm[lc] as f64;
                if budget > 0.0 && margin.is_finite() {
                    ((budget - margin) / budget).clamp(0.0, 1.0) * REFERENCE_MARGIN_PENALTY
                } else {
                    0.0
                }
            }
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        };
        let high_speed_edge_penalty = match class {
            NetClassKind::Hv | NetClassKind::Signal => {
                self.high_speed_edge_field[c] as f64 * HIGH_SPEED_EDGE_PENALTY
            }
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        };
        let inner_ground_penalty = match class {
            NetClassKind::Hv | NetClassKind::Signal => {
                self.inner_dual_ground_field[lc] as f64 * INNER_DUAL_GROUND_PENALTY
            }
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        };
        let high_speed_spacing_penalty = match class {
            NetClassKind::Hv | NetClassKind::Signal => {
                self.high_speed_spacing_field[lc] as f64 * HIGH_SPEED_SPACING_PENALTY
                    + self.high_speed_adjacent_layer_spacing_field[lc] as f64
                        * HIGH_SPEED_SPACING_PENALTY
            }
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        };
        let power_reference_penalty = match class {
            NetClassKind::Hv | NetClassKind::Signal => {
                self.power_reference_field[lc] as f64 * POWER_REFERENCE_PENALTY
            }
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        };
        1.0 + self.creepage_weight * hazard
            + self.layer_affinity(layer, class)
            + self.top_side_affinity(layer, class)
            + reference_penalty
            + reference_margin_penalty
            + inner_ground_penalty
            + high_speed_spacing_penalty
            + power_reference_penalty
            + high_speed_edge_penalty
    }

    fn via_cost(&self, class: NetClassKind) -> f64 {
        match class {
            NetClassKind::Hv | NetClassKind::Signal => self.via_cost * HIGH_SPEED_VIA_MULTIPLIER,
            NetClassKind::Power | NetClassKind::Ground => self.via_cost,
        }
    }
}
