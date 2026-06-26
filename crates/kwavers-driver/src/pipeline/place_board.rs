//! Placement → routing bridge: build the [`Board`] from placed components, plus the placement-stage keepout/repulsion helpers the loop drives.

use std::collections::HashMap;

use crate::board::{Board, NetId, Pad};
use crate::geom::Nm;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::route::grid::NodeId;
use crate::route::pathfinder::PadObstacle;
use crate::route::NetTerminals;
use crate::rules::DesignRules;

/// Pad pitch (centre-to-centre) at or below which a footprint cannot escape on the top layer and must
/// fan out via-in-pad to an inner layer ([`FootprintDef::needs_escape`]). Derived from the holoHV rule
/// set: a top-layer channel between two adjacent pads must fit a signal track (0.15 mm) plus two
/// clearances (2 × 0.13 mm) ⇒ ≥ 0.41 mm of pad-edge gap; with a typical fine-pitch pad ≈ 0.3 mm wide
/// that puts the direct-routable pitch at ≈ 0.71 mm. Parts finer than this (QFN 0.5, QSOP 0.635,
/// fine-QFP 0.5/0.65) escape downward; coarser parts (TQFP 0.8, SOIC 1.27, passives) route on top.
const FINE_PITCH_ESCAPE: crate::geom::Nm = crate::geom::Nm(700_000);

/// Distribute multiple instances of the same active-IC footprint into a regular grid spanning the
/// usable board area. Called once before the cooptimize round loop so the annealer starts from a
/// well-spread initial placement rather than all ICs clustered at their seed position.
///
/// Only movable `ActiveIc` components are redistributed; locked components and non-IC roles
/// keep their caller-specified positions. Groups with a single member are unchanged.
pub(super) fn seed_symmetric_groups(
    comps: &mut [Component],
    lib: &[FootprintDef],
    movable: &[usize],
    cfg: &crate::place::PlaceConfig,
) {
    use crate::geom::Point;
    use crate::place::footprint::Role;
    use std::collections::BTreeMap;

    let w = cfg.board.0.to_mm();
    let h = cfg.board.1.to_mm();
    let m = cfg.margin.to_mm();

    // Group movable ActiveIc indices by footprint index so identical parts share a spread grid.
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &ci in movable {
        if matches!(lib[comps[ci].fp].role, Role::ActiveIc) {
            groups.entry(comps[ci].fp).or_default().push(ci);
        }
    }

    for indices in groups.values() {
        let n = indices.len();
        if n < 2 {
            continue;
        }
        // Grid dimensions: as close to square as possible.
        let cols = ((n as f64).sqrt().ceil() as usize).max(1);
        let rows = n.div_ceil(cols);
        let usable_w = (w - 2.0 * m).max(0.0);
        let usable_h = (h - 2.0 * m).max(0.0);
        for (k, &ci) in indices.iter().enumerate() {
            let col = k % cols;
            let row = k / cols;
            // Place at the centroid of cell (col, row) in the usable-area grid.
            let x = m + usable_w * (col as f64 + 0.5) / cols as f64;
            let y = m + usable_h * (row as f64 + 0.5) / rows as f64;
            comps[ci].placement.pos = Point::new(Nm::from_mm(x), Nm::from_mm(y));
        }
    }
}

pub(super) fn apply_emi_pair_repulsion(
    board: &crate::board::Board,
    comps: &mut [Component],
    lib: &[FootprintDef],
    movable: &[usize],
    cfg: &crate::place::PlaceConfig,
    coupling: Nm,
) {
    let movable_set: std::collections::BTreeSet<usize> = movable.iter().copied().collect();
    let coupling_nm = coupling.0 as f64;
    let mut delta = vec![(0.0f64, 0.0f64); comps.len()];
    for i in 0..comps.len() {
        for j in (i + 1)..comps.len() {
            let mut worst = 0.0f64;
            for (pi, _layers_i, ni) in comps[i].placed_pads(lib) {
                let Some(ni) = ni else { continue };
                let hi = matches!(board.class_of(ni), crate::board::NetClassKind::Hv);
                for (pj, _layers_j, nj) in comps[j].placed_pads(lib) {
                    let Some(nj) = nj else { continue };
                    let hj = matches!(board.class_of(nj), crate::board::NetClassKind::Hv);
                    if hi == hj {
                        continue;
                    }
                    worst = worst.max((coupling_nm - pi.euclid(pj)).max(0.0));
                }
            }
            if worst <= 0.0 {
                continue;
            }
            let ci = comps[i].placement.pos;
            let cj = comps[j].placement.pos;
            let vx = (ci.x.0 - cj.x.0) as f64;
            let vy = (ci.y.0 - cj.y.0) as f64;
            let mag = (vx * vx + vy * vy).sqrt();
            let (ux, uy) = if mag > 1.0 {
                (vx / mag, vy / mag)
            } else {
                (1.0, 0.0)
            };
            let step_mm = worst * 0.5e-6;
            if movable_set.contains(&i) {
                delta[i].0 += ux * step_mm;
                delta[i].1 += uy * step_mm;
            }
            if movable_set.contains(&j) {
                delta[j].0 -= ux * step_mm;
                delta[j].1 -= uy * step_mm;
            }
        }
    }
    for idx in movable {
        let (dx, dy) = delta[*idx];
        if dx == 0.0 && dy == 0.0 {
            continue;
        }
        comps[*idx].placement.pos = crate::geom::Point::new(
            comps[*idx].placement.pos.x + Nm::from_mm(dx),
            comps[*idx].placement.pos.y + Nm::from_mm(dy),
        );
        clamp_component_inside(&mut comps[*idx], lib, cfg);
    }
}

fn clamp_component_inside(
    c: &mut Component,
    lib: &[FootprintDef],
    cfg: &crate::place::PlaceConfig,
) {
    let rect = c.courtyard(lib);
    let margin = cfg.margin;
    let min_x = margin + Nm(c.placement.pos.x.0 - rect.min.x.0);
    let max_x = cfg.board.0 - margin - Nm(rect.max.x.0 - c.placement.pos.x.0);
    let min_y = margin + Nm(c.placement.pos.y.0 - rect.min.y.0);
    let max_y = cfg.board.1 - margin - Nm(rect.max.y.0 - c.placement.pos.y.0);
    c.placement.pos.x = Nm(c.placement.pos.x.0.clamp(min_x.0, max_x.0));
    c.placement.pos.y = Nm(c.placement.pos.y.0.clamp(min_y.0, max_y.0));
}

/// Block the top-layer cells under each component's body — the courtyard interior, minus a one-cell
/// ring around the component's own pads (the escape region). Keeps top-layer copper out of package
/// bodies / exposed thermal pads; inner and bottom layers remain free to route underneath.
pub(super) fn block_component_bodies(
    grid: &mut crate::route::Grid,
    comps: &[Component],
    lib: &[FootprintDef],
    spec: crate::geom::GridSpec,
    rules: &DesignRules,
) {
    use crate::route::grid::NodeId;
    for c in comps {
        let fp = &lib[c.fp];
        let rect = c.courtyard(lib);
        let hw = Nm((rect.max.x - rect.min.x).0 / 2);
        let hh = Nm((rect.max.y - rect.min.y).0 / 2);
        // A part that escapes via-in-pad downward (a BGA, or any fine-pitch QFN/QSOP/QFP) has its
        // whole top-layer footprint as a keepout — nothing routes over the pad field on top (otherwise
        // a track shorts an unconnected pad, since every cell is near a pad and the normal
        // pad-exemption would block nothing).
        let escape = (fp.ball_pitch.is_some()
            || (fp.needs_escape(FINE_PITCH_ESCAPE)
                && rules.via_policy == crate::rules::ViaPolicy::Hdi))
            && spec.nlayers > 1;
        let pad_cells: Vec<(usize, usize)> = if escape {
            Vec::new()
        } else {
            (0..fp.pads.len())
                .map(|k| spec.cell_of(c.pad_pos(lib, k)))
                .collect()
        };
        for (ix, iy) in spec.cells_in_rect(c.placement.pos, hw, hh) {
            let near_pad = pad_cells.iter().any(|&(px, py)| {
                (px as i64 - ix as i64).abs() <= 1 && (py as i64 - iy as i64).abs() <= 1
            });
            if !near_pad {
                grid.block(NodeId(spec.node_index(ix, iy, 0)));
            }
        }
    }
}

/// Block routing cells under every board mechanical feature (fiducials, mounting holes) on all
/// layers, by its keepout radius. Shares [`crate::io::mechanical_features`] with emission so the
/// reserved holes are exactly the drilled holes.
pub(super) fn block_mechanical(grid: &mut crate::route::Grid, spec: crate::geom::GridSpec) {
    use crate::route::grid::NodeId;
    let w = (spec.nx as i64 - 1) as f64 * spec.pitch.0 as f64 / 1.0e6;
    let h = (spec.ny as i64 - 1) as f64 * spec.pitch.0 as f64 / 1.0e6;
    for f in crate::io::mechanical_features(w, h) {
        let r = Nm::from_mm(f.keepout_mm());
        let center = crate::geom::Point::new(Nm::from_mm(f.x), Nm::from_mm(f.y));
        for (ix, iy) in spec.cells_in_rect(center, r, r) {
            for layer in 0..spec.nlayers {
                grid.block(NodeId(spec.node_index(ix, iy, layer)));
            }
        }
    }
}

/// Router inputs derived from a placement.
pub struct RoutingInputs {
    /// Per-net logical terminals and their accessible grid nodes.
    pub terminals: Vec<NetTerminals>,
    /// Per-pad foreign-clearance halos.
    pub obstacles: Vec<PadObstacle>,
}

/// Record placed pads on `board` and build the router terminals + pad-clearance halos.
///
/// The halo half-extent is
/// `pad_half + min_clearance + widest_track/2 + clearance_guard`: the guard absorbs
/// grid-to-continuous-geometry error between the router's centreline lattice and KiCad's exact
/// segment-to-pad DRC.
pub fn place_to_board(
    board: &mut Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> RoutingInputs {
    let spec = board.spec;
    // Track halo: clearance + half the widest track + a continuous-geometry guard. The guard term
    // covers two distinct rounding effects:
    // (1) Centreline-grid snap: a pad centre not on the routing grid means the nearest grid cell
    //     is ≤ pitch/2 away, placing routed track centrelines that distance from the pad centre
    //     rather than exactly at it — the guard compensates for this sub-grid residual.
    // (2) Diagonal-approach at pad corners: a diagonal track whose centre is `pitch` away in
    //     each axis (grid-legal) comes sqrt(2)×pitch/2 closer to a pad *corner* than a
    //     perpendicular approach to the pad *edge*. On a 0.5 mm grid with 0.25 mm tracks the
    //     worst-case edge-to-edge gap shrinks to ≈ 0.087 mm (below the 0.13 mm rule). Setting
    //     geometry_guard = 0.07 mm (up from 0.05 mm) extends the rectangular obstacle halo by
    //     0.02 mm per side so the diagonal cell is blocked and the physical gap ≥ 0.13 mm.
    let geometry_guard = Nm::from_mm(0.07);
    let pad_margin_signal = rules.min_clearance + Nm(rules.signal_track.0 / 2) + geometry_guard;
    let pad_margin_hv = rules.min_clearance + Nm(rules.hv_track.0 / 2) + geometry_guard;
    let pad_margin_power = rules.min_clearance + Nm(rules.power_track.0 / 2) + geometry_guard;
    // Via keepout (larger): clearance + via radius — only *via placement* needs this wider margin
    // (a via's annular ring is bigger than a track), so it doesn't choke track channels.
    let via_margin = rules.min_clearance + Nm(rules.via_diameter().0 / 2) + geometry_guard;
    let mut terminals: HashMap<NetId, Vec<Vec<NodeId>>> = HashMap::new();
    let mut obstacles: Vec<PadObstacle> = Vec::new();

    for c in comps {
        let fp = &lib[c.fp];
        for (k, pad) in fp.pads.iter().enumerate() {
            let pos = c.pad_pos(lib, k);
            let net = c.nets[k];
            board.add_pad(Pad {
                pos,
                layers: pad.layers.clone(),
                net,
            });

            // --- Pad-clearance obstacle (built for EVERY pad, including unconnected copper) --------
            // A pad is physical copper: foreign nets must clear it, and a track/via that crossed an
            // *unconnected* pad (net = None) would short to dead copper — so the obstacle is built
            // whether or not the pad carries a net. A **thru-hole** pad is a full-stack barrel, so its
            // halo blocks *every* layer; an **SMD** pad is single-layer copper, so it blocks only the
            // layer it sits on (an SMD top pad does not block inner-layer tracks under it). The via
            // keepout always spans all layers (a via is a full-stack drill).
            let (pw, ph) = c.placement.rot.apply_size(pad.size);
            let through = pad.layers.len() > 1; // thru-hole barrel spans the whole stack
            let build_nodes = |margin: Nm| {
                let mut nodes = Vec::new();
                for (cx, cy) in
                    spec.cells_in_rect(pos, Nm(pw.0 / 2) + margin, Nm(ph.0 / 2) + margin)
                {
                    if through {
                        for layer in 0..spec.nlayers {
                            nodes.push(NodeId(spec.node_index(cx, cy, layer)));
                        }
                    } else {
                        for &layer in &pad.layers {
                            nodes.push(NodeId(spec.node_index(cx, cy, layer.0 as usize)));
                        }
                    }
                }
                nodes
            };
            let is_pad_hv = net
                .map(|n| matches!(board.class_of(n), crate::board::NetClassKind::Hv))
                .unwrap_or(false);
            let pad_margin_creepage = Nm::from_mm(0.60);

            let nodes_signal = build_nodes(pad_margin_signal);
            let nodes_signal_creepage = if is_pad_hv {
                build_nodes(pad_margin_creepage)
            } else {
                nodes_signal.clone()
            };

            let nodes_hv = build_nodes(pad_margin_hv);
            let nodes_hv_creepage = if !is_pad_hv {
                build_nodes(pad_margin_creepage)
            } else {
                nodes_hv.clone()
            };

            let nodes_power = build_nodes(pad_margin_power);
            let nodes_power_creepage = if is_pad_hv {
                build_nodes(pad_margin_creepage)
            } else {
                nodes_power.clone()
            };

            let mut via_keepout = Vec::new();
            for (cx, cy) in
                spec.cells_in_rect(pos, Nm(pw.0 / 2) + via_margin, Nm(ph.0 / 2) + via_margin)
            {
                for layer in 0..spec.nlayers {
                    via_keepout.push(NodeId(spec.node_index(cx, cy, layer)));
                }
            }
            let is_large_pad = pw.to_mm() * ph.to_mm() > 16.0;
            let nodes_shrunken = build_nodes(geometry_guard);
            let same_component_nets: Vec<NetId> = c.nets.iter().filter_map(|&n| n).collect();

            obstacles.push(PadObstacle {
                net, // None ⇒ keepout for every net (unconnected copper)
                same_component_nets,
                nodes_signal,
                nodes_signal_creepage,
                nodes_hv,
                nodes_hv_creepage,
                nodes_power,
                nodes_power_creepage,
                via_keepout,
                drilled: pad.layers.len() > 1,
                nodes_shrunken,
                is_large_pad,
                is_hv_pad: is_pad_hv,
            });

            // --- Terminal + escape via (netted pads only) -----------------------------------------
            let Some(n) = net else { continue };
            // Escape layer: a BGA ball is buried in the array and can't be reached on the congested
            // top layer, so it fans out *via-in-pad* to the first inner layer where the router
            // escapes it through the channels between the ball vias. Ordinary (leaded/QFN/connector)
            // pads route directly on the top layer.
            let (ix, iy) = spec.cell_of(pos);
            let terminal_group = if through {
                (0..spec.nlayers)
                    .map(|layer| NodeId(spec.node_index(ix, iy, layer)))
                    .collect()
            } else if (fp.ball_pitch.is_some()
                || (fp.needs_escape(FINE_PITCH_ESCAPE)
                    && rules.via_policy == crate::rules::ViaPolicy::Hdi))
                && spec.nlayers > 1
            {
                let inner = 1usize;
                // A BGA/fine-pitch escape is a **via-in-pad, plated over (VIPPO)** — filled and capped
                // so the ball pad stays solderable. Its construction follows the board's via policy:
                // a filled through-hole on a standard stackup, a laser micro-via on an HDI stackup.
                let (from, to, kind, drill, diameter) =
                    rules.resolve_via(0, inner as u16, spec.nlayers);
                board.vias.push(crate::board::Via {
                    // Snap the fanout via to the terminal's grid cell so it coincides with any
                    // routing via the escape uses (no sub-cell offset that reads as hole-to-hole).
                    pos: spec.point_of(ix, iy),
                    drill,
                    diameter,
                    net: n,
                    from,
                    to,
                    kind,
                    filled: true,
                });
                vec![NodeId(spec.node_index(ix, iy, inner))]
            } else {
                let layer = pad.layers.first().map_or(0, |layer| layer.0 as usize);
                vec![NodeId(spec.node_index(ix, iy, layer))]
            };
            terminals.entry(n).or_default().push(terminal_group);
        }
    }

    // Sort by net id so the routing order is deterministic (HashMap iteration order is randomised
    // per run; without this the negotiated-congestion result — and its emitted geometry — varies).
    let mut terminals: Vec<NetTerminals> = terminals
        .into_iter()
        .map(|(net, t)| NetTerminals {
            net,
            class: board.class_of(net),
            terminal_groups: t,
        })
        .collect();
    terminals.sort_by_key(|t| t.net.0);

    RoutingInputs {
        terminals,
        obstacles,
    }
}
