//! Functional-region cohesion, board utilization, assembly-alignment, LV↔HV isolation-barrier
//! drift, and congestion feed-back energy terms.
//!
//! Extracted from `compute::energy()` — lines 426–697 of the original `compute.rs`.
//! All arithmetic is bit-for-bit identical to the original; no logic was changed.

use std::collections::{BTreeMap, BTreeSet};

use super::config::{Axis, CongestionField, EnergyTerms, PlaceConfig};
use super::geom::rotation_axis;
use crate::board::NetId;
use crate::geom::Nm;
use crate::place::component::Component;
use crate::place::footprint::{FootprintDef, IsolationDomain, Role};

/// Grouping key for a functional region.
///
/// Regions are keyed by the semantic relationship that caused them to be created:
/// * `AssociatedMain` — components sharing the same explicitly-associated main IC.
/// * `SignalNet` — components sharing a non-power, non-global signal net.
/// * `RailDomain` — components sharing ≥2 power-rail nets (same power domain).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum RegionKey {
    AssociatedMain(usize),
    SignalNet(NetId),
    RailDomain(Vec<NetId>),
}

/// Axis-aligned bounding box + member set for a functional placement region.
struct FunctionalRegion {
    members: BTreeSet<usize>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    block_min_x: f64,
    block_min_y: f64,
    block_max_x: f64,
    block_max_y: f64,
}

/// Insert or expand a functional region to include component `idx`.
fn add_region_member(
    regions: &mut BTreeMap<RegionKey, FunctionalRegion>,
    key: RegionKey,
    idx: usize,
    c: &Component,
    lib: &[FootprintDef],
) {
    let x = c.placement.pos.x.to_mm();
    let y = c.placement.pos.y.to_mm();
    let courtyard = c.courtyard(lib);
    let block_min_x = courtyard.min.x.to_mm();
    let block_min_y = courtyard.min.y.to_mm();
    let block_max_x = courtyard.max.x.to_mm();
    let block_max_y = courtyard.max.y.to_mm();
    regions
        .entry(key)
        .and_modify(|r| {
            r.members.insert(idx);
            r.min_x = r.min_x.min(x);
            r.min_y = r.min_y.min(y);
            r.max_x = r.max_x.max(x);
            r.max_y = r.max_y.max(y);
            r.block_min_x = r.block_min_x.min(block_min_x);
            r.block_min_y = r.block_min_y.min(block_min_y);
            r.block_max_x = r.block_max_x.max(block_max_x);
            r.block_max_y = r.block_max_y.max(block_max_y);
        })
        .or_insert_with(|| FunctionalRegion {
            members: BTreeSet::from([idx]),
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
            block_min_x,
            block_min_y,
            block_max_x,
            block_max_y,
        });
}

/// Accumulate functional-region cohesion, board-utilization, assembly-alignment,
/// LV↔HV isolation-barrier drift, and congestion penalty terms.
///
/// Returns the congestion weight extracted from `congestion` (0.0 if `None`), which the
/// orchestrator uses when forming the weighted total.
///
/// # Arguments
/// * `t` — energy accumulator; `regional`, `utilization`, `alignment`, `isolation_drift`, and
///   `congestion` are updated.
/// * `comps` — full component list.
/// * `lib` — footprint library.
/// * `cfg` — placement configuration.
/// * `congestion` — optional routing-congestion field from a prior routing pass.
/// * `half_clear` — half the courtyard clearance (unused directly; retained for API symmetry).
/// * `w` / `h` — board width/height in mm.
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_floorplan(
    t: &mut EnergyTerms,
    comps: &[Component],
    lib: &[FootprintDef],
    cfg: &PlaceConfig,
    congestion: Option<&CongestionField>,
    _half_clear: Nm,
    w: f64,
    h: f64,
) -> f64 {
    let m = cfg.margin.to_mm();

    // Functional regions: group components sharing local signal/control nets, components sharing
    // the same power-pin rail domain, and support parts explicitly associated with a main IC so
    // schematic subsections remain spatially coherent. Nets touching every component are treated as
    // global distribution/return nets and ignored here; their copper is still handled by
    // HPWL/PDN/routing. Components from another region placed inside a local region add an
    // intrusion-depth penalty, matching the guide's rule to avoid mixing unrelated functional
    // groups into another block.
    let mut regions: BTreeMap<RegionKey, FunctionalRegion> = BTreeMap::new();
    for (idx, c) in comps.iter().enumerate() {
        if let Some(ic) = c.assoc_ic.filter(|ic| *ic < comps.len()) {
            add_region_member(
                &mut regions,
                RegionKey::AssociatedMain(ic),
                ic,
                &comps[ic],
                lib,
            );
            add_region_member(&mut regions, RegionKey::AssociatedMain(ic), idx, c, lib);
        }
        let mut seen = BTreeSet::new();
        let mut rail_domain = BTreeSet::new();
        let fp = &lib[c.fp];
        for (pad_idx, net) in c.nets.iter().enumerate() {
            let Some(n) = *net else { continue };
            if fp.pads[pad_idx].power_pin {
                rail_domain.insert(n);
                continue;
            }
            if seen.insert(n) {
                add_region_member(&mut regions, RegionKey::SignalNet(n), idx, c, lib);
            }
        }
        if rail_domain.len() >= 2 {
            add_region_member(
                &mut regions,
                RegionKey::RailDomain(rail_domain.into_iter().collect()),
                idx,
                c,
                lib,
            );
        }
    }
    for (key, region) in regions {
        if region.members.len() >= 2 && region.members.len() < comps.len() {
            t.regional += (region.max_x - region.min_x) + (region.max_y - region.min_y);
            for (idx, c) in comps.iter().enumerate() {
                if region.members.contains(&idx) {
                    continue;
                }
                let x = c.placement.pos.x.to_mm();
                let y = c.placement.pos.y.to_mm();
                let courtyard = c.courtyard(lib);
                let c_min_x = courtyard.min.x.to_mm();
                let c_min_y = courtyard.min.y.to_mm();
                let c_max_x = courtyard.max.x.to_mm();
                let c_max_y = courtyard.max.y.to_mm();
                let overlap_x = region.block_max_x.min(c_max_x) - region.block_min_x.max(c_min_x);
                let overlap_y = region.block_max_y.min(c_max_y) - region.block_min_y.max(c_min_y);
                if overlap_x > 0.0 && overlap_y > 0.0 {
                    let center_depth = if (region.min_x..=region.max_x).contains(&x)
                        && (region.min_y..=region.max_y).contains(&y)
                    {
                        (x - region.min_x)
                            .min(region.max_x - x)
                            .min(y - region.min_y)
                            .min(region.max_y - y)
                            .max(0.0)
                    } else {
                        0.0
                    };
                    t.regional += 1.0 + center_depth + overlap_x.min(overlap_y);
                }
                if matches!(key, RegionKey::SignalNet(_)) && matches!(lib[c.fp].role, Role::Power) {
                    let isolation = (cfg.courtyard_clearance.to_mm() * 2.0).max(1.0);
                    let halo_min_x = region.block_min_x - isolation;
                    let halo_min_y = region.block_min_y - isolation;
                    let halo_max_x = region.block_max_x + isolation;
                    let halo_max_y = region.block_max_y + isolation;
                    let halo_overlap_x = halo_max_x.min(c_max_x) - halo_min_x.max(c_min_x);
                    let halo_overlap_y = halo_max_y.min(c_max_y) - halo_min_y.max(c_min_y);
                    if halo_overlap_x > 0.0 && halo_overlap_y > 0.0 {
                        t.regional += halo_overlap_x.min(halo_overlap_y) / isolation;
                    }
                }
            }
        }
    }

    // Board utilization: sample a fixed 3x3 macro grid inside the keep-in area and penalise the
    // average distance to the nearest movable, non-connector component centre. Locked edge
    // connectors are mechanical constraints, not functional area coverage; counting them lets a
    // stack connector or programming header mask an off-centre movable cluster.
    if !comps.is_empty() {
        let utilization_comps: Vec<&Component> = comps
            .iter()
            .filter(|c| !c.locked && !matches!(lib[c.fp].role, Role::Connector))
            .collect();
        let utilization_comps: Vec<&Component> = if utilization_comps.is_empty() {
            comps.iter().collect()
        } else {
            utilization_comps
        };
        let xs = [m + (w - 2.0 * m) * 0.2, w / 2.0, m + (w - 2.0 * m) * 0.8];
        let ys = [m + (h - 2.0 * m) * 0.2, h / 2.0, m + (h - 2.0 * m) * 0.8];
        for sx in xs {
            for sy in ys {
                let nearest = utilization_comps
                    .iter()
                    .map(|c| {
                        let dx = c.placement.pos.x.to_mm() - sx;
                        let dy = c.placement.pos.y.to_mm() - sy;
                        (dx * dx + dy * dy).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min);
                t.utilization += nearest;
            }
        }
        t.utilization /= 9.0;
    }

    // Assembly/routing alignment: similar components with the same footprint and role should share
    // the same 0/180 vs 90/270 axis. Pin-1 polarity is intentionally not penalised so half-turn
    // passive flips remain available for short escape routing.
    for (i, a) in comps.iter().enumerate() {
        let afp = &lib[a.fp];
        for b in comps.iter().skip(i + 1) {
            let bfp = &lib[b.fp];
            if a.fp == b.fp
                && afp.role == bfp.role
                && rotation_axis(a.placement.rot) != rotation_axis(b.placement.rot)
            {
                t.alignment += 1.0;
            }
        }
    }

    // LV↔HV isolation-barrier drift: each non-locked component pays penalty for its *projection*
    // onto the configured `isolation_axis`. LV components park on the axis-min edge (lowest
    // coordinate on the chosen axis); HV components park on the axis-max edge (board extent on
    // that axis). The penalty is `proj` (low) vs. `axis_max - proj` (high) in mm, so the annealer
    // has a smooth gradient nudging misplaced components across the barrier instead of a hard
    // wall. Locked components are exempt (mechanical / inter-tile constraints always win).
    //
    // Both-domain gate: the term is silent when the board contains *only* LV or *only* HV parts.
    // An accidental all-LV design with `weights.isolation_drift > 0` would otherwise degenerate
    // every component toward x=0 (a useless strip layout), and an all-HV design would degenerate
    // toward x=board.width. The gate restores the term's intent: pull *misplaced* components
    // across the barrier toward their domain's expected edge.
    let mut has_lv = false;
    let mut has_hv = false;
    for c in comps {
        if c.locked {
            continue;
        }
        match c.isolation_domain {
            IsolationDomain::Lv => has_lv = true,
            IsolationDomain::Hv => has_hv = true,
        }
    }
    if has_lv && has_hv {
        let axis_max_along = match cfg.isolation_axis {
            Axis::X => w,
            Axis::Y => h,
        };
        for c in comps {
            if c.locked {
                continue;
            }
            let proj = match cfg.isolation_axis {
                Axis::X => c.placement.pos.x.to_mm(),
                Axis::Y => c.placement.pos.y.to_mm(),
            };
            match c.isolation_domain {
                IsolationDomain::Lv => {
                    t.isolation_drift += proj.max(0.0);
                }
                IsolationDomain::Hv => {
                    t.isolation_drift += (axis_max_along - proj).max(0.0);
                }
            }
        }
    }

    // Congestion: sum the fed-back congestion at every pad cell, so the placer moves pin-dense
    // components out of regions the router struggled with.
    let mut cong_weight = 0.0;
    if let Some(cg) = congestion {
        cong_weight = cg.weight;
        for c in comps {
            for (pos, _layers, _net) in c.placed_pads(lib) {
                t.congestion += cg.at(pos);
            }
        }
    }

    cong_weight
}
