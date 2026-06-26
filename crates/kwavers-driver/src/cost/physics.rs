//! [`PhysicsCost`] — the physics- and manufacturing-guided cost model.
//!
//! See the [module-level documentation](super) for the full cost-seam rationale + the
//! proximity-hazard-model summary.
//!
//! `PhysicsCost` is the concrete implementation of the `RoutingCost` trait that folds
//! high-voltage creepage, layer affinity, high-speed edge clearance, and reference-plane quality
//! into the base cost. The struct holds *precomputed fields* — arrays sized by `nx * ny *
//! nlayers` that aggregate per-cell penalties once at construction. `PhysicsCost::new` does that
//! aggregation by leaning on the `geometry_modulated` penalty kernels (`adjacent_reference_margin`,
//! `high_speed_track_proximity`, `high_speed_adjacent_layer_track_proximity`, sum-weighted
//! `proximity`).
//!
//! The `impl RoutingCost for PhysicsCost` bridge in `adapter.rs` reads these precomputed fields
//! and applies the per-class penalty weights at query time.
//!
//! (Plain backticks rather than `[`X`](path)` intra-doc links: the four kernels are
//! `pub(super)` and rustdoc's `private_intra_doc_links` lint rejects clickable links to them
//! from this public module's docstring.)

use super::geometry_modulated::{
    adjacent_reference_margin, high_speed_adjacent_layer_track_proximity,
    high_speed_track_proximity, proximity,
};
use crate::board::{Board, LayerId, NetClassKind};
use crate::geom::{point_in_polygon, GridSpec, Point};
use crate::rules::{CreepageRule, DesignRules};

// Per-class penalty constants. Owned by `PhysicsCost` (the struct that interprets them) and
// shared with `adapter.rs` (the trait bridge that applies them at query time). `pub(super)`
// visibility scopes them to `crate::cost` and its children — not part of the crate's public
// API surface (no re-export) — so the constants are mutable-internal to the cost slice.
pub(super) const REFERENCE_PLANE_PENALTY: f64 = 4.0;
pub(super) const REFERENCE_MARGIN_PENALTY: f64 = 2.0;
pub(super) const INNER_DUAL_GROUND_PENALTY: f64 = 4.0;
pub(super) const POWER_REFERENCE_PENALTY: f64 = 2.0;
pub(super) const HIGH_SPEED_EDGE_PENALTY: f64 = 3.0;
pub(super) const HIGH_SPEED_SPACING_PENALTY: f64 = 1.5;
pub(super) const HIGH_SPEED_VIA_MULTIPLIER: f64 = 2.0;
pub(super) const HIGH_SPEED_BOTTOM_LAYER_PENALTY: f64 = 0.5;

/// Physics- and manufacturing-guided cost.
///
/// Two spatial penalty fields are precomputed over the grid's in-plane cells:
/// * `lv_field[c]` rises as cell `c` approaches a **low-voltage** pad — charged to HV nets so they
///   keep creepage distance from LV copper.
/// * `hv_field[c]` rises as cell `c` approaches a **high-voltage** pad — charged to LV nets,
///   symmetrically.
///
/// Layer affinity adds a smaller term: HV prefers outer copper (creepage is a surface phenomenon,
/// easier to police on an outer layer), control prefers inner copper (shielded between planes), and
/// high-speed outer-layer routing prefers the top side over the bottom side.
#[derive(Debug, Clone)]
pub struct PhysicsCost {
    pub(super) spec: GridSpec,
    pub(super) lv_field: Vec<f32>,
    pub(super) hv_field: Vec<f32>,
    pub(super) reference_field: Vec<f32>,
    pub(super) reference_margin_nm: Vec<f32>,
    pub(super) inner_dual_ground_field: Vec<f32>,
    pub(super) power_reference_field: Vec<f32>,
    pub(super) high_speed_edge_field: Vec<f32>,
    pub(super) high_speed_spacing_field: Vec<f32>,
    pub(super) high_speed_adjacent_layer_spacing_field: Vec<f32>,
    pub(super) signal_reference_margin_budget: f64,
    pub(super) hv_reference_margin_budget: f64,
    pub(super) creepage_weight: f64,
    pub(super) affinity_weight: f64,
    pub(super) via_cost: f64,
}

impl PhysicsCost {
    /// Build the cost field from a board's pads and design rules.
    ///
    /// `creepage_weight` scales the HV/LV proximity hazard; `affinity_weight` scales the
    /// layer-preference term. Both are pure search weights (no physical unit) — they bias the
    /// route, while the *legality* of the result is judged by congestion (capacity) and, downstream,
    /// by DRC.
    #[must_use]
    pub fn new(
        spec: GridSpec,
        board: &Board,
        rules: &DesignRules,
        creepage: CreepageRule,
        creepage_weight: f64,
        affinity_weight: f64,
    ) -> Self {
        let plane = spec.nx * spec.ny;
        let mut lv_field = vec![0.0f32; plane];
        let mut hv_field = vec![0.0f32; plane];
        let mut reference_field = vec![0.0f32; plane * spec.nlayers];
        let mut reference_margin_nm = vec![f32::INFINITY; plane * spec.nlayers];
        let mut inner_dual_ground_field = vec![0.0f32; plane * spec.nlayers];
        let mut power_reference_field = vec![0.0f32; plane * spec.nlayers];
        let mut high_speed_edge_field = vec![0.0f32; plane];
        let mut high_speed_spacing_field = vec![0.0f32; plane * spec.nlayers];
        let mut high_speed_adjacent_layer_spacing_field = vec![0.0f32; plane * spec.nlayers];
        let creep = creepage.hv_clearance.0 as f64;
        let high_speed_edge = rules.high_speed_edge_clearance.0 as f64;
        let signal_reference_margin_budget =
            rules.high_speed_reference_plane_margin_widths * rules.signal_track.0 as f64;
        let hv_reference_margin_budget =
            rules.high_speed_reference_plane_margin_widths * rules.hv_track.0 as f64;
        let max_x = spec.origin.x.0 + (spec.nx as i64 - 1) * spec.pitch.0;
        let max_y = spec.origin.y.0 + (spec.ny as i64 - 1) * spec.pitch.0;

        // Collect domain pad centres once.
        let mut lv_pads: Vec<Point> = Vec::new();
        let mut hv_pads: Vec<Point> = Vec::new();
        for pad in &board.pads {
            let class = pad.net.map(|n| board.class_of(n));
            match class {
                Some(NetClassKind::Hv) => hv_pads.push(pad.pos),
                Some(c) if c.is_low_voltage() => lv_pads.push(pad.pos),
                _ => {}
            }
        }

        for iy in 0..spec.ny {
            for ix in 0..spec.nx {
                let c = iy * spec.nx + ix;
                let p = spec.point_of(ix, iy);
                lv_field[c] = proximity(p, &lv_pads, creep);
                hv_field[c] = proximity(p, &hv_pads, creep);
                let edge_dist = (p.x.0 - spec.origin.x.0)
                    .min(p.y.0 - spec.origin.y.0)
                    .min(max_x - p.x.0)
                    .min(max_y - p.y.0)
                    .max(0) as f64;
                high_speed_edge_field[c] = if high_speed_edge > 0.0 {
                    ((high_speed_edge - edge_dist) / high_speed_edge).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                for layer in 0..spec.nlayers {
                    let has_adjacent_ground = board.zones.iter().any(|zone| {
                        zone.layer.0.abs_diff(layer as u16) == 1
                            && board.class_of(zone.net) == NetClassKind::Ground
                            && point_in_polygon(p, &zone.polygon)
                    });
                    let has_adjacent_power = board.zones.iter().any(|zone| {
                        zone.layer.0.abs_diff(layer as u16) == 1
                            && board.class_of(zone.net) == NetClassKind::Power
                            && point_in_polygon(p, &zone.polygon)
                    });
                    let has_reference = has_adjacent_ground || has_adjacent_power;
                    reference_field[layer * plane + c] = if has_reference { 0.0 } else { 1.0 };
                    reference_margin_nm[layer * plane + c] =
                        adjacent_reference_margin(board, p, layer)
                            .map(|margin| margin as f32)
                            .unwrap_or(f32::INFINITY);
                    power_reference_field[layer * plane + c] =
                        if has_adjacent_power && !has_adjacent_ground {
                            1.0
                        } else {
                            0.0
                        };
                    let has_dual_ground = layer > 0
                        && layer + 1 < spec.nlayers
                        && [layer - 1, layer + 1].iter().all(|&adjacent| {
                            board.zones.iter().any(|zone| {
                                zone.layer.0 == adjacent as u16
                                    && board.class_of(zone.net) == NetClassKind::Ground
                                    && point_in_polygon(p, &zone.polygon)
                            })
                        });
                    inner_dual_ground_field[layer * plane + c] =
                        if layer == 0 || layer + 1 >= spec.nlayers || has_dual_ground {
                            0.0
                        } else {
                            1.0
                        };
                    high_speed_spacing_field[layer * plane + c] =
                        high_speed_track_proximity(p, LayerId(layer as u16), board, rules);
                    high_speed_adjacent_layer_spacing_field[layer * plane + c] =
                        high_speed_adjacent_layer_track_proximity(
                            p,
                            LayerId(layer as u16),
                            board,
                            rules,
                        );
                }
            }
        }

        PhysicsCost {
            spec,
            lv_field,
            hv_field,
            reference_field,
            reference_margin_nm,
            inner_dual_ground_field,
            power_reference_field,
            high_speed_edge_field,
            high_speed_spacing_field,
            high_speed_adjacent_layer_spacing_field,
            signal_reference_margin_budget,
            hv_reference_margin_budget,
            creepage_weight,
            affinity_weight,
            via_cost: 10.0,
        }
    }

    /// Layer preference penalty: HV prefers outer copper; control prefers inner copper. Queried
    /// at adapter time via [`impl RoutingCost for PhysicsCost`](super::adapter).
    pub(super) fn layer_affinity(&self, layer: LayerId, class: NetClassKind) -> f64 {
        let l = layer.0 as usize;
        let last = self.spec.nlayers.saturating_sub(1);
        let is_outer = l == 0 || l == last;
        match class {
            // HV prefers outer copper; charge inner layers.
            NetClassKind::Hv => {
                if is_outer {
                    0.0
                } else {
                    self.affinity_weight
                }
            }
            // Control prefers inner copper; charge outer layers.
            NetClassKind::Signal => {
                if is_outer {
                    self.affinity_weight
                } else {
                    0.0
                }
            }
            // Power/ground are plane-like: no layer preference.
            NetClassKind::Power | NetClassKind::Ground => 0.0,
        }
    }

    /// Top-side preference (outer-bottom penalty) for high-speed outer-layer routing. The
    /// bottom-side penalty is a fraction ([`HIGH_SPEED_BOTTOM_LAYER_PENALTY`]) of the base
    /// affinity weight so high-speed routing prefers the top side when forced to outer layers.
    /// Queried at adapter time.
    pub(super) fn top_side_affinity(&self, layer: LayerId, class: NetClassKind) -> f64 {
        let l = layer.0 as usize;
        let last = self.spec.nlayers.saturating_sub(1);
        if l == last && matches!(class, NetClassKind::Hv | NetClassKind::Signal) {
            self.affinity_weight * HIGH_SPEED_BOTTOM_LAYER_PENALTY
        } else {
            0.0
        }
    }
}
