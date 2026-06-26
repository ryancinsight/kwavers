//! The adversarial place↔route co-optimization loop ([`cooptimize`]) and its min-layer / min-area variants.

use crate::board::{Board, NetId};
use crate::geom::Nm;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use crate::route::grid::NodeId;
use crate::rules::DesignRules;

use super::config::CoOpt;
use super::place_board::{
    apply_emi_pair_repulsion, block_component_bodies, block_mechanical, place_to_board,
    seed_symmetric_groups,
};
use super::result::CoOptResult;

/// Courtyard area (mm²) of a footprint — the "charge" deposited by the component-density field
/// ([`crate::physics::thermal::solve_board`] sourced by area instead of dissipation). A larger
/// part pushes harder on its neighbours, exactly as a larger charge does in the electrostatic
/// density-equalisation analogy of analytical placement.
fn footprint_area_mm2(fp: &FootprintDef) -> f64 {
    let (w, h) = fp.courtyard;
    w.to_mm() * h.to_mm()
}

/// Distinct copper layers carrying tracks or vias on a board.
fn layers_used(board: &crate::board::Board) -> usize {
    let mut seen = std::collections::BTreeSet::new();
    for t in &board.tracks {
        seen.insert(t.layer.0);
    }
    for v in &board.vias {
        seen.insert(v.from.0);
        seen.insert(v.to.0);
    }
    seen.len()
}

/// Count different-net copper occupying the same routing grid cell/layer, or different-net vias
/// occupying the same via column. This is the discrete routing-grid hard fault that complements the
/// continuous-geometry clearance audit after DFM rewriting.
#[must_use]
pub(super) fn grid_occupancy_shorts(board: &Board) -> usize {
    use std::collections::HashMap;

    let mut occupied: HashMap<(usize, usize, u16), NetId> = HashMap::new();
    let spec = board.spec;
    let mut shorts = 0;
    let mut mark = |occupied: &mut HashMap<(usize, usize, u16), NetId>,
                    point: crate::geom::Point,
                    layer: u16,
                    net: NetId| {
        let (ix, iy) = spec.cell_of(point);
        match occupied.insert((ix, iy, layer), net) {
            Some(previous) if previous != net => shorts += 1,
            _ => {}
        }
    };
    for track in &board.tracks {
        mark(&mut occupied, track.start, track.layer.0, track.net);
        mark(&mut occupied, track.end, track.layer.0, track.net);
    }

    let mut via_columns: HashMap<(usize, usize), NetId> = HashMap::new();
    for via in &board.vias {
        let (ix, iy) = spec.cell_of(via.pos);
        match via_columns.insert((ix, iy), via.net) {
            Some(previous) if previous != via.net => shorts += 1,
            _ => {}
        }
    }

    shorts
}

/// Find the **fewest layers** that route the design complete + legal, trying `layer_options` in
/// ascending order (4-layer is the cheap commodity tier; only escalate when a design needs it).
/// Returns `None` when `layer_options` is empty; otherwise the first clean result, or the last
/// attempt if none is fully clean.
pub fn cooptimize_min_layers(
    nets_template: &crate::board::Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
    cfg: &CoOpt,
    layer_options: &[usize],
) -> Option<CoOptResult> {
    let mut last: Option<CoOptResult> = None;
    for &nl in layer_options {
        let mut tmpl = nets_template.clone();
        tmpl.spec.nlayers = nl;
        let r = cooptimize(&tmpl, comps.to_vec(), lib, rules, cfg);
        // Accept the fewest layers that are not just complete+legal but hard-DRC/assembly-clean.
        // Soft adversarial metrics (flight-line crossings, near-shorts) still influence `risk_score`
        // but do not block a manufacturing-clean layer decision.
        if r.complete
            && r.legal
            && r.report.hard_drc_clean()
            && component_clearance_clean(&r, lib, cfg)
        {
            return Some(r);
        }
        // Keep the best fallback by (complete, legal, assembly, hard DRC, lowest risk).
        let better = last
            .as_ref()
            .map(|p| {
                (
                    r.complete,
                    r.legal,
                    component_clearance_clean(&r, lib, cfg),
                    r.report.hard_drc_clean(),
                    std::cmp::Reverse((r.report.risk_score * 1000.0) as i64),
                ) > (
                    p.complete,
                    p.legal,
                    component_clearance_clean(p, lib, cfg),
                    p.report.hard_drc_clean(),
                    std::cmp::Reverse((p.report.risk_score * 1000.0) as i64),
                )
            })
            .unwrap_or(true);
        if better {
            last = Some(r);
        }
    }
    last
}

/// Find the **smallest board** that routes clean, trying `sizes` smallest-area first and returning
/// the first that is complete + legal + manufacturing-clean (no via-adjacency or dangling ends).
/// Returns `None` when `sizes` is empty; otherwise the first clean result, or the best attempt
/// (complete, then legal, then lowest risk) if none is fully clean.
///
/// Routability is non-monotonic in board size (a smaller board can route while a slightly larger one
/// gets an unlucky placement), so a smallest-first scan — rather than a binary search — is the robust
/// way to pin the minimum: it returns the genuine smallest clean fit and never accepts a dirty board
/// when a smaller clean one exists earlier in the scan. This auto-sizes the board instead of relying
/// on a hand-tuned dimension (hardening the placement-seed sensitivity of a fixed size). Each size is
/// a fresh routing grid at the template's pitch/layer count carrying the same nets. Falls back to the
/// best attempt (complete, then legal, then lowest risk) if none is fully clean.
pub fn cooptimize_min_area(
    nets_template: &crate::board::Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
    cfg: &CoOpt,
    sizes: &[(Nm, Nm)],
) -> Option<CoOptResult> {
    let mut order: Vec<(Nm, Nm)> = sizes.to_vec();
    order.sort_by_key(|(w, h)| w.0.saturating_mul(h.0));
    let mut best: Option<CoOptResult> = None;
    for (w, h) in order {
        let Ok(spec) = crate::geom::GridSpec::cover(
            w,
            h,
            nets_template.spec.pitch,
            nets_template.spec.nlayers,
        ) else {
            continue; // degenerate size
        };
        let mut tmpl = Board::new(spec);
        for n in &nets_template.nets {
            tmpl.add_net(n.name.clone(), n.class);
        }
        let mut c = *cfg;
        c.place.board = (w, h);
        let r = cooptimize(&tmpl, comps.to_vec(), lib, rules, &c);
        if r.complete
            && r.legal
            && r.report.hard_drc_clean()
            && component_clearance_clean(&r, lib, &c)
        {
            return Some(r); // smallest clean board
        }
        let better = best
            .as_ref()
            .map(|p| {
                (
                    r.complete,
                    r.legal,
                    component_clearance_clean(&r, lib, &c),
                    r.report.hard_drc_clean(),
                    std::cmp::Reverse(r.report.risk_score as i64),
                ) > (
                    p.complete,
                    p.legal,
                    component_clearance_clean(p, lib, &c),
                    p.report.hard_drc_clean(),
                    std::cmp::Reverse(p.report.risk_score as i64),
                )
            })
            .unwrap_or(true);
        if better {
            best = Some(r);
        }
    }
    best
}

/// Run the adversarial place↔route co-optimization loop: each round places (biased away from the
/// previous round's congestion + critic weakness), routes, and is attacked by the critic; the best
/// round (complete, then legal, then lowest risk) is kept. Stops early when the best stops
/// improving (`patience`). This is the reusable core both design tiles drive.
pub fn cooptimize(
    nets_template: &crate::board::Board,
    mut comps: Vec<Component>,
    lib: &[FootprintDef],
    rules: &DesignRules,
    cfg: &CoOpt,
) -> CoOptResult {
    use crate::cost::PhysicsCost;
    use crate::place::{anneal, CongestionField};
    use crate::route::{Grid, Router};

    let spec = nets_template.spec;
    // Locked components (e.g. inter-tile mating connectors) keep their fixed placement — only the
    // rest are handed to the annealer to move.
    let movable: Vec<usize> = (0..comps.len()).filter(|&i| !comps[i].locked).collect();
    let mut feedback: Option<CongestionField> = None;
    // Ranking: hard routing/manufacturing predicates first; then, when enabled, physics judge terms
    // such as peak temperature and EMI coupling; then soft routing risk. This makes physics guidance
    // a real objective instead of only a placement bias.
    type Score = (
        bool,
        bool,
        bool,
        bool,
        bool,
        std::cmp::Reverse<i64>,
        std::cmp::Reverse<i64>,
        std::cmp::Reverse<i64>,
        std::cmp::Reverse<i64>,
    );
    let mut best: Option<(
        crate::board::Board,
        Vec<Component>,
        crate::audit::FaultReport,
        Score,
    )> = None;
    let mut stale = 0usize;
    let mut rounds_run = 0;

    // Pre-seed: distribute identical active ICs across the board before round 0 so the first
    // routing pass starts from a well-spread placement, not a clumped pile.
    if cfg.seed_groups {
        seed_symmetric_groups(&mut comps, lib, &movable, &cfg.place);
    }

    for round in 0..=cfg.rounds {
        rounds_run += 1;
        if round > 0 && cfg.emi_weight > 0.0 {
            apply_emi_pair_repulsion(
                nets_template,
                &mut comps,
                lib,
                &movable,
                &cfg.place,
                Nm::from_mm(6.0),
            );
        }
        if round > 0 {
            anneal(
                &mut comps,
                lib,
                &cfg.place,
                &movable,
                &cfg.anneal,
                feedback.as_ref(),
            );
        }

        let mut rb = nets_template.clone();
        let inputs = place_to_board(&mut rb, &comps, lib, rules);
        let cost = PhysicsCost::new(
            spec,
            &rb,
            rules,
            cfg.creepage,
            cfg.creepage_weight,
            cfg.affinity_weight,
        );
        let mut grid = Grid::new(spec);
        grid.set_diagonal_routing(rules.diagonal_routing);
        // Width-aware clearance halo: on a fine grid whose pitch is below the clearance quantum
        // (signal track + clearance), one-net-per-cell no longer holds clearance, so the router must
        // keep foreign copper `ceil((track + clearance)/pitch) − 1` cells away. Derived from the
        // *signal* track (the dominant fine-pitch case): on a coarse grid (≥ ~0.28 mm) this is 0, so
        // the halo is inert and coarse-grid routing is unchanged.
        let sep = rules.signal_track.0 + rules.min_clearance.0;
        let clearance_cells = (sep + spec.pitch.0 - 1) / spec.pitch.0; // ceil(sep / pitch), ≥ 1
        grid.set_clearance_radius((clearance_cells - 1).max(0) as usize);
        grid.set_via_clearance_limit(rules.via_diameter() + rules.min_clearance);
        let widest_track = rules
            .signal_track
            .0
            .max(rules.hv_track.0)
            .max(rules.power_track.0);
        grid.reserve_edge(crate::geom::Nm(rules.edge_clearance.0 + widest_track / 2));
        // Component-body keepout: block the *top* layer under each package body (courtyard cells
        // not adjacent to one of the component's own pads). Tracks then route around the body on the
        // component layer, or under it on inner/bottom layers — never through the body/thermal pad.
        block_component_bodies(&mut grid, &comps, lib, spec, rules);
        // Mechanical keepout: reserve the fiducial pads and mounting-hole barrels (all layers) so no
        // track runs into a hole/fiducial — critical once the board is tight enough that these land
        // inside the routable area.
        block_mechanical(&mut grid, spec);
        // Register pre-placed BGA fanout vias (via-in-pad escapes from `place_to_board`) so the
        // router treats each ball's column as that net's via and routes around foreign ones.
        for v in &rb.vias {
            let (ix, iy) = spec.cell_of(v.pos);
            grid.set_via(NodeId(spec.node_index(ix, iy, 0)), v.net.0 as i32);
        }
        let mut router = Router::new(grid, cost).with_params(crate::route::PathFinderParams {
            max_iter: cfg.pathfinder_max_iter,
            ..Default::default()
        });
        let outcome = router.route_with_obstacles(&inputs.terminals, &inputs.obstacles);
        router.apply_to_board(&mut rb, &inputs.terminals, &outcome, rules);
        // DFM: merge any BGA fanout via that coincides with the router's via at the same ball cell,
        // then consolidate the cell-by-cell segments into one per straight run — fewer photoplot
        // vertices / acid-trap sites, copper geometry unchanged.
        crate::dfm::dedup_vias(&mut rb, rules);
        crate::dfm::merge_collinear(&mut rb);
        // Router-level diagonal-crossing guards already prevent opposed foreign diagonals and
        // via-corner clips. Preserve clean 45-degree routes, and only orthogonalize the diagonal
        // segments that actually form acute-angle DFM traps.
        crate::dfm::chamfer_diagonal_traps(&mut rb);
        crate::dfm::merge_collinear(&mut rb); // consolidate duplicate legs the chamfer may create
                                              // Convert any remaining diagonals that violate geometric clearance to foreign-net vias.
                                              // The router's grid-level corner-cell guards prevent most cases; this pass closes the gap
                                              // for sub-grid-level perpendicular distance violations that KiCad DRC detects.
        crate::dfm::resolve_diagonal_via_clearance(&mut rb, rules);
        crate::dfm::merge_collinear(&mut rb); // merge new orthogonal legs with collinear neighbours
                                              // Replace 90° L-corners with mitered 135° bends. Per IPC-2221 §10.4.3 only angles < 90°
                                              // are acid-trap concerns; converting right-angle corners to 135° eliminates their
                                              // false-positive contribution to `sharp_bends` and improves Gerber geometry.
                                              // A pad-proximity guard inside the pass prevents miter endpoints landing in foreign-pad
                                              // clearance halos (root-cause fix for DFM-pass-induced clearance violations).
        crate::dfm::miter_right_angle_corners(&mut rb, rules.signal_track, &comps, lib, rules);
        crate::dfm::split_track_body_junctions(&mut rb);
        crate::dfm::pad_entry_stubs(&mut rb, &comps, lib, rules);
        crate::dfm::split_track_body_junctions(&mut rb);
        // Delete orphan copper (pad-less islands a rip-up/re-route left behind): floating vias/stubs
        // that connect nothing, register as `via_dangling`, and can short/mask-bridge a foreign
        // feature they overlap (e.g. a fiducial). Connectivity-safe — every net's pads are on other
        // islands. Run after the pad-entry stubs so real pad connections are anchored first.
        crate::dfm::remove_orphan_copper(&mut rb);

        let report = crate::audit::audit(&rb, &comps, lib, rules);
        let cong = router.grid().congestion_field();
        let weak = crate::audit::weakness_field(&report, spec, 1.0);
        // Physics guidance: solve this placement's steady-state thermal field and fold its hotspots
        // into the feedback so the next placement spreads dissipative parts (flattens the field). The
        // heat sources come from `cfg.dissipation_w` — the derived pulser loss model for a driver tile
        // (so hot series resistors are spread too), or the coarse role estimate by default.
        let tfield = crate::physics::thermal::solve_board(
            spec,
            &comps,
            lib,
            cfg.dissipation_w,
            20.0,
            1.6e-3,
            10.0,
            150,
        );
        let tpeak = tfield.peak();
        let thermal_per_column: Vec<f32> = if tpeak > 0.0 {
            tfield
                .temp
                .iter()
                .map(|rise| (rise / tpeak) as f32)
                .collect()
        } else {
            vec![0.0; spec.nx * spec.ny]
        };
        // EMI guidance: HV↔LV pad-proximity zones (coupling within ~6 mm) fed back so the next
        // placement separates the switching node from sensitive control.
        let emi_points = crate::audit::emi_hotspots(&rb, Nm::from_mm(6.0));
        let emi = crate::audit::rasterize_hotspots_radius(&emi_points, spec, Nm::from_mm(3.0), 1.0);
        // Density guidance: diffuse every component's courtyard *area* through the same Poisson
        // solver as the thermal field, giving a potential that peaks where parts cluster. Folded
        // into the feedback it spreads the whole BOM to fill the board (ePlace area-as-charge),
        // so no per-design `thermal_spacing` padding is needed to relax a dense central cluster.
        let dfield = crate::physics::thermal::solve_board(
            spec,
            &comps,
            lib,
            footprint_area_mm2,
            20.0,
            1.6e-3,
            10.0,
            150,
        );
        let dpeak = dfield.peak();
        let density_per_column: Vec<f32> = if dpeak > 0.0 {
            dfield.temp.iter().map(|d| (d / dpeak) as f32).collect()
        } else {
            vec![0.0; spec.nx * spec.ny]
        };
        let fw = cfg.feedback_weight as f32;
        let tw = cfg.thermal_weight as f32;
        let ew = cfg.emi_weight as f32;
        let dw = cfg.density_weight as f32;
        let per_column: Vec<f32> = (0..cong.len())
            .map(|i| {
                cong[i] * fw
                    + weak.per_column[i] * 40.0 * fw
                    + thermal_per_column[i] * tw
                    + emi.per_column[i] * ew
                    + density_per_column[i] * dw
            })
            .collect();
        feedback = Some(CongestionField {
            spec,
            per_column,
            weight: 1.0,
        });

        // Thermal is a judge objective only when guidance is enabled (`thermal_weight > 0`); at 0
        // the loop is fully thermal-blind (clean ablation), so the term collapses to a constant.
        let tpeak_term = if cfg.thermal_weight > 0.0 {
            std::cmp::Reverse((tfield.peak() * 1000.0) as i64)
        } else {
            std::cmp::Reverse(0)
        };
        let emi_term = if cfg.emi_weight > 0.0 {
            std::cmp::Reverse(emi_points.len() as i64)
        } else {
            std::cmp::Reverse(0)
        };
        // Density is the **final** tiebreaker (ranked below risk): among placements that are equally
        // hard-DRC-clean and equally low-risk, prefer the one whose component-area field is flattest
        // (most spread). On a real board, congestion already shows up in `risk_score`, so this rarely
        // overrides it — it just resolves the ties a trivial-routing layout leaves, and makes the
        // spreading a genuine selected objective rather than only a placement bias. Constant at 0
        // when density guidance is off (clean ablation).
        let density_term = if cfg.density_weight > 0.0 {
            std::cmp::Reverse((dfield.peak() * 1000.0) as i64)
        } else {
            std::cmp::Reverse(0)
        };
        // **Verified** completeness, not the router's claim: extract the actual copper connectivity
        // from the emitted geometry (tracks + vias + pads) and require zero opens. The router's
        // `outcome.complete` only asserts every terminal was *reached during search*; a subsequent
        // rip-up/re-route can leave a net broken into separate copper islands (a stub that connects
        // nothing), which `outcome.complete` would still call done. Deriving `complete` from `lvs`
        // makes it unmaskable — a net whose pads are not on one electrical island is incomplete,
        // full stop. (Opens only; shorts are the separate legality/DRC terms below.)
        let connected = crate::verify::lvs(&rb).opens.is_empty();
        let score: Score = (
            connected,
            outcome.legal,
            placement_clearance_clean(&comps, lib, cfg.place.courtyard_clearance),
            grid_occupancy_shorts(&rb) == 0,
            report.hard_drc_clean(),
            emi_term,
            tpeak_term,
            std::cmp::Reverse((report.risk_score * 1000.0) as i64),
            density_term,
        );
        let improved = best.as_ref().map(|(_, _, _, s)| score > *s).unwrap_or(true);
        if improved {
            best = Some((rb, comps.clone(), report, score));
            stale = 0;
        } else {
            stale += 1;
            if stale >= cfg.patience {
                break;
            }
        }
    }

    let (board, comps, report, score) = best.expect(
        "invariant: cfg.rounds >= 1 (CoOpt::default() sets 4; callers must not set rounds=0)",
    );
    let used = layers_used(&board);
    CoOptResult {
        board,
        comps,
        report,
        legal: score.1,
        complete: score.0,
        rounds_run,
        layer_count: spec.nlayers,
        layers_used: used,
    }
}

fn placement_clearance_clean(comps: &[Component], lib: &[FootprintDef], clearance: Nm) -> bool {
    crate::place::component::component_clearance_violations(comps, lib, clearance).is_empty()
}

pub(super) fn component_clearance_clean(
    r: &CoOptResult,
    lib: &[FootprintDef],
    cfg: &CoOpt,
) -> bool {
    placement_clearance_clean(&r.comps, lib, cfg.place.courtyard_clearance)
}
