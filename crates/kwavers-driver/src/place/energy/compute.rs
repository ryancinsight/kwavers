//! Placement energy computation — the single `energy()` entry point.

use std::collections::{BTreeMap, BTreeSet};

use crate::board::NetId;
use crate::geom::{segments_cross, Nm, Point};
use crate::place::component::{is_crystal_refdes, is_surge_suppressor_refdes, Component, Rect};
use crate::place::footprint::{FootprintDef, Role};
use super::config::{Axis, CongestionField, EnergyTerms, PlaceConfig};
use super::geom::{
    carries_connected_signal, connector_ingress_unit, has_non_power_pad_on_net,
    nearest_board_edge_point, non_power_signal_net_count, rect_gap_mm, rotation_axis,
    segment_intersects_rect,
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
    let mech_keepouts: Vec<Rect> = crate::io::mechanical_features(w, h)
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

    // Regional EMI floorplanning: connectors radiate and couple external noise, so sensitive
    // high-speed ICs with connected signal pads keep a clearance halo from connector courtyards.
    // The hard edge keepout still handles board-edge impedance; this term handles connector
    // proximity inside the floorplan.
    let connector_emi_clearance = (cfg.courtyard_clearance.to_mm() * 2.0).max(1.0);
    for active_ic in comps {
        let fp = &lib[active_ic.fp];
        if !matches!(fp.role, Role::ActiveIc) || !carries_connected_signal(active_ic, fp) {
            continue;
        }
        let active_rect = active_ic.courtyard(lib);
        for connector in comps {
            if !matches!(lib[connector.fp].role, Role::Connector) {
                continue;
            }
            let gap = rect_gap_mm(active_rect, connector.courtyard(lib));
            t.regional += (connector_emi_clearance - gap).max(0.0);
        }
    }

    // Decoupling: each bypass cap to the nearest power pin of its associated IC.
    for c in comps {
        if !matches!(lib[c.fp].role, Role::Decoupling) {
            continue;
        }
        let Some(ic) = c.assoc_ic else { continue };
        let cap = c.placement.pos;
        let ic_c = &comps[ic];
        let mut nearest = f64::INFINITY;
        for (k, pad) in lib[ic_c.fp].pads.iter().enumerate() {
            if pad.power_pin {
                let d = cap.euclid(ic_c.pad_pos(lib, k)) * 1.0e-6;
                nearest = nearest.min(d);
            }
        }
        if nearest.is_finite() {
            t.decoupling += nearest;
        }
    }

    // Termination: resistor-like passives on a net should sit at the active pad they terminate,
    // not merely somewhere on the net's HPWL box. The final audit applies the high-speed-specific
    // 2 mm budget; this term gives placement a continuous objective before that hard gate.
    for terminator in comps {
        if terminator.fp >= lib.len()
            || !matches!(lib[terminator.fp].role, Role::Passive)
            || !terminator.refdes.starts_with('R')
        {
            continue;
        }
        let mut nearest = f64::INFINITY;
        for (term_pos, _term_layers, term_net) in terminator.placed_pads(lib) {
            let Some(net) = term_net else { continue };
            for active in comps {
                if active.fp >= lib.len() || !matches!(lib[active.fp].role, Role::ActiveIc) {
                    continue;
                }
                for (active_pos, _active_layers, active_net) in active.placed_pads(lib) {
                    if active_net == Some(net) {
                        nearest = nearest.min(term_pos.euclid(active_pos) * 1.0e-6);
                    }
                }
            }
        }
        if nearest.is_finite() {
            t.termination += nearest;
        }
    }

    // Surge/ESD suppressors: diode-like passives on incoming connector nets should sit at the
    // connector before the protected trace enters the board. This keeps the clamp path short and
    // reduces the chance that the connector-to-clamp segment needs a parasitic via.
    for suppressor in comps {
        if suppressor.fp >= lib.len()
            || !matches!(lib[suppressor.fp].role, Role::Passive)
            || !is_surge_suppressor_refdes(&suppressor.refdes)
        {
            continue;
        }
        let mut nearest = f64::INFINITY;
        for (supp_pos, _supp_layers, supp_net) in suppressor.placed_pads(lib) {
            let Some(net) = supp_net else { continue };
            for connector in comps {
                if connector.fp >= lib.len() || !matches!(lib[connector.fp].role, Role::Connector) {
                    continue;
                }
                for (conn_pos, _conn_layers, conn_net) in connector.placed_pads(lib) {
                    if conn_net == Some(net) {
                        nearest = nearest.min(supp_pos.euclid(conn_pos) * 1.0e-6);
                    }
                }
            }
        }
        if nearest.is_finite() {
            t.regional += nearest;
        }
    }

    // Crystal/resonator support: clock-source components associated with a main IC must sit close
    // to the shared clock pins so those critical routes are short before controlled-impedance
    // routing consumes the remaining channels.
    for oscillator in comps {
        if oscillator.fp >= lib.len() || !is_crystal_refdes(&oscillator.refdes) {
            continue;
        }
        let Some(ic_idx) = oscillator.assoc_ic.filter(|ic| *ic < comps.len()) else {
            continue;
        };
        let ic = &comps[ic_idx];
        let mut nearest = f64::INFINITY;
        for (osc_pos, _osc_layers, osc_net) in oscillator.placed_pads(lib) {
            let Some(net) = osc_net else { continue };
            for (ic_pos, _ic_layers, ic_net) in ic.placed_pads(lib) {
                if ic_net == Some(net) {
                    nearest = nearest.min(osc_pos.euclid(ic_pos) * 1.0e-6);
                }
            }
        }
        if nearest.is_finite() {
            t.regional += nearest;
        }
    }

    // HPWL: per-net bounding-box half-perimeter. BTreeMap (not HashMap) so the f64 summation
    // order is deterministic — otherwise randomised iteration perturbs the total and flips SA
    // accept/reject decisions, making placement non-reproducible.
    let mut bbox: BTreeMap<NetId, (f64, f64, f64, f64)> = BTreeMap::new();
    for c in comps {
        for (pos, _layers, net) in c.placed_pads(lib) {
            if let Some(n) = net {
                let x = pos.x.to_mm();
                let y = pos.y.to_mm();
                let e = bbox.entry(n).or_insert((x, y, x, y));
                e.0 = e.0.min(x);
                e.1 = e.1.min(y);
                e.2 = e.2.max(x);
                e.3 = e.3.max(y);
            }
        }
    }
    for (_n, (x1, y1, x2, y2)) in bbox {
        t.hpwl += (x2 - x1) + (y2 - y1);
    }

    let mut net_centers: BTreeMap<NetId, Vec<(usize, Point)>> = BTreeMap::new();
    for (idx, c) in comps.iter().enumerate() {
        // Each component typically has 2–8 pads across 1–4 nets; a Vec linear scan is
        // faster than a BTreeMap allocation for this small N (no heap per-node overhead).
        let mut sum_by_net: Vec<(NetId, (i64, i64, i64))> = Vec::new();
        for (pos, _layers, net) in c.placed_pads(lib) {
            if let Some(n) = net {
                if let Some(e) = sum_by_net.iter_mut().find(|(id, _)| *id == n) {
                    e.1.0 += pos.x.0;
                    e.1.1 += pos.y.0;
                    e.1.2 += 1;
                } else {
                    sum_by_net.push((n, (pos.x.0, pos.y.0, 1)));
                }
            }
        }
        // Sort by NetId before inserting so net_centers iteration order matches the outer
        // BTreeMap's determinism requirement (same insertion order → same HPWL sum).
        sum_by_net.sort_unstable_by_key(|(id, _)| *id);
        for (net, (sx, sy, count)) in sum_by_net {
            net_centers
                .entry(net)
                .or_default()
                .push((idx, Point::new(Nm(sx / count), Nm(sy / count))));
        }
    }
    struct FlightLine {
        members: [usize; 2],
        a: Point,
        b: Point,
    }
    let mut flight_lines: Vec<FlightLine> = Vec::new();
    let mut local_flow_vectors: Vec<Vec<(f64, f64)>> = vec![Vec::new(); comps.len()];
    for (net, centers) in &net_centers {
        if centers.len() == 2 {
            let a = centers[0].0;
            let b = centers[1].0;
            let a_fp = &lib[comps[a].fp];
            let b_fp = &lib[comps[b].fp];
            if matches!(a_fp.role, Role::ActiveIc)
                && matches!(b_fp.role, Role::ActiveIc)
                && non_power_signal_net_count(&comps[a], a_fp) == 1
                && non_power_signal_net_count(&comps[b], b_fp) == 1
                && has_non_power_pad_on_net(&comps[a], a_fp, *net)
                && has_non_power_pad_on_net(&comps[b], b_fp, *net)
            {
                t.regional += centers[0].1.euclid(centers[1].1) * 1.0e-6;
            }
            let ax = comps[a].placement.pos.x.to_mm();
            let ay = comps[a].placement.pos.y.to_mm();
            let bx = comps[b].placement.pos.x.to_mm();
            let by = comps[b].placement.pos.y.to_mm();
            local_flow_vectors[a].push((bx - ax, by - ay));
            local_flow_vectors[b].push((ax - bx, ay - by));
            flight_lines.push(FlightLine {
                members: [centers[0].0, centers[1].0],
                a: centers[0].1,
                b: centers[1].1,
            });
            let a_connector = matches!(lib[comps[a].fp].role, Role::Connector);
            let b_connector = matches!(lib[comps[b].fp].role, Role::Connector);
            if a_connector != b_connector {
                let (connector, other) = if a_connector { (a, b) } else { (b, a) };
                let cx = comps[connector].placement.pos.x.to_mm();
                let cy = comps[connector].placement.pos.y.to_mm();
                let ox = comps[other].placement.pos.x.to_mm();
                let oy = comps[other].placement.pos.y.to_mm();
                let dx = ox - cx;
                let dy = oy - cy;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.0 {
                    let (ux, uy) = connector_ingress_unit(comps[connector].placement.pos, w, h);
                    let cosine = ((dx * ux + dy * uy) / len).clamp(-1.0, 1.0);
                    t.regional += 1.0 - cosine;
                }
            }
        }
    }
    // Regional signal flow: at a component joining two local point-to-point nets, both neighbors on
    // the same side form a fold-back and orthogonal neighbors form a dogleg instead of the smooth
    // unidirectional flow expected from a floorplanned high-speed subsection. The normalized dot
    // product makes a straight fold-back cost 1.0, and the normalized cross product makes a right
    // angle cost 1.0 while collinear through-flow costs 0.
    for vectors in &local_flow_vectors {
        for i in 0..vectors.len() {
            let (ax, ay) = vectors[i];
            let a_len = (ax * ax + ay * ay).sqrt();
            for &(bx, by) in vectors.iter().skip(i + 1) {
                let b_len = (bx * bx + by * by).sqrt();
                if a_len == 0.0 || b_len == 0.0 {
                    continue;
                }
                let direction_cosine = (ax * bx + ay * by) / (a_len * b_len);
                if direction_cosine > 0.0 {
                    t.regional += direction_cosine;
                }
                t.regional += ((ax * by - ay * bx) / (a_len * b_len)).abs();
            }
        }
    }
    for i in 0..flight_lines.len() {
        for other in flight_lines.iter().skip(i + 1) {
            let line = &flight_lines[i];
            if segments_cross(line.a, line.b, other.a, other.b) {
                t.flow_crossing += 1.0;
            }
        }
        let line = &flight_lines[i];
        for (idx, c) in comps.iter().enumerate() {
            if line.members.contains(&idx) {
                continue;
            }
            let channel_keepout = c.courtyard(lib).inflate(half_clear);
            if segment_intersects_rect(line.a, line.b, channel_keepout) {
                t.channel_blockage += 1.0;
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum RegionKey {
        AssociatedMain(usize),
        SignalNet(NetId),
        RailDomain(Vec<NetId>),
    }

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
            crate::place::footprint::IsolationDomain::Lv => has_lv = true,
            crate::place::footprint::IsolationDomain::Hv => has_hv = true,
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
                crate::place::footprint::IsolationDomain::Lv => {
                    t.isolation_drift += proj.max(0.0);
                }
                crate::place::footprint::IsolationDomain::Hv => {
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
