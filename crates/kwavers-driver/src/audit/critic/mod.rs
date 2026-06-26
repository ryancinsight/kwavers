//! Board-level DFM / physics critic.
//!
//! The single public entry point is [`audit`], which calls all per-family detectors
//! and accumulates results into a [`FaultReport`] with a weighted risk score.
//! Hotspot rasterisation and efficiency-audit helpers for the placement feedback
//! loop are also exposed here.

pub mod congestion;
pub mod diagnostics;

pub use congestion::{rasterize_hotspots, rasterize_hotspots_radius, weakness_field};
pub use diagnostics::{
    charge_recycling_efficiency_audit, ChargeRecyclingReport, pulse_skip_interference_audit,
    PulseSkipInterferenceReport,
};

use crate::board::Board;
use crate::geom::{Nm, Point};
use crate::place::{Component, FootprintDef};
use crate::rules::DesignRules;
use crate::verify::{parasitic_ac_coupling_check, schematic_isolation_bfs};
use crate::audit::fault_report::FaultReport;
use crate::audit::detect_power::*;
use crate::audit::{antenna::*, crosstalk::*, detect_diff_pair::*, detect_high_speed::*, detect_track::*, shorts::*};

/// Run the full adversarial manufacturing and physics audit over a routed board.
#[must_use]
pub fn audit(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    rules: &DesignRules,
) -> FaultReport {
    let build_up_mm = rules.build_up_mm;
    let (crossings, mut hotspots) = flight_crossings(board);
    let (clearance, _clearance_score, clearance_pts) = near_shorts(board, rules.min_clearance);
    let (near, near_score, near_pts) = near_shorts(board, rules.min_clearance * 3);
    let (dangling, dang_pts) = dangling_ends(board);
    // Couple tracks separated by up to ~1.5 cells: adjacent (one-cell-gap) parallel runs are the
    // crosstalk-prone case; the threshold must exceed the grid pitch to register them.
    let crosstalk_n = crosstalk(board, Nm(board.spec.pitch.0 * 3 / 2));
    let (via_adj, via_pts) = via_adjacency(board, rules.via_diameter() + rules.min_clearance);
    let traps = acid_traps(board);
    let via_count = board.vias.len();
    hotspots.extend(near_pts);
    hotspots.extend(clearance_pts);
    hotspots.extend(dang_pts);
    hotspots.extend(via_pts);

    let (sharp_bends, sharp_pts) = detect_sharp_bends(board);
    let (track_crossing_violations, track_crossing_pts) = detect_track_crossing_violations(board);
    let (hole_clearance_violations, hole_clearance_pts) =
        detect_hole_clearance_violations(board, rules);
    let (serpentine_spacing, serpentine_length, serpentine_compensation_distance, serpentine_pts) =
        detect_serpentine_violations(board, rules);
    let (via_spacing, via_spacing_pts) = detect_via_spacing_violations(board, rules);
    let (plane_hotspot_via_spacing, plane_hotspot_via_spacing_pts) =
        detect_plane_hotspot_via_spacing_violations(board, rules);
    let (
        diff_pair_violations,
        diff_pair_layer_mismatch_violations,
        diff_pair_via_count_violations,
        diff_pair_length_mismatch_violations,
        diff_pair_segment_length_mismatch_violations,
        diff_pair_spacing_variation_violations,
        diff_pair_via_symmetry_violations,
        diff_pair_total_length_mismatch_mm,
        diff_pair_pts,
    ) = detect_diff_pair_violations(board, comps, lib, rules);
    let (
        diff_pair_interface_layer_mismatch_violations,
        diff_pair_interface_via_count_mismatch_violations,
        diff_pair_interface_mismatch_pts,
    ) = detect_diff_pair_interface_mismatch_violations(board);
    let (diff_pair_coupling_cap_symmetry_violations, diff_pair_coupling_cap_symmetry_pts) =
        detect_diff_pair_coupling_cap_symmetry_violations(board, comps, rules);
    let (diff_pair_coupling_cap_package_violations, diff_pair_coupling_cap_package_pts) =
        detect_diff_pair_coupling_cap_package_violations(board, comps, lib, rules);
    let (diff_pair_stitching_cap_symmetry_violations, diff_pair_stitching_cap_symmetry_pts) =
        detect_diff_pair_stitching_cap_symmetry_violations(board, comps, rules);
    let (diff_pair_pad_entry_mismatch_violations, diff_pair_pad_entry_mismatch_pts) =
        detect_diff_pair_pad_entry_mismatch_violations(board, rules);
    let (diff_pair_pad_entry_length_violations, diff_pair_pad_entry_length_pts) =
        detect_diff_pair_pad_entry_length_violations(board, rules);
    let (parallel_bus_length_mismatch_violations, parallel_bus_length_mismatch_pts) =
        detect_parallel_bus_length_mismatch_violations(board, rules);
    let (diff_pair_keepout_violations, diff_pair_keepout_pts) =
        detect_diff_pair_keepout_violations(board, rules);
    let (high_speed_edge_violations, high_speed_edge_pts) =
        detect_high_speed_edge_violations(board, rules);
    let (high_speed_component_edge_violations, high_speed_component_edge_pts) =
        detect_high_speed_component_edge_violations(board, comps, lib, rules);
    let (high_speed_termination_placement_violations, high_speed_termination_placement_pts) =
        detect_high_speed_termination_placement_violations(board, comps, lib, rules);
    let (high_speed_parallel_spacing_violations, high_speed_parallel_spacing_pts) =
        detect_high_speed_parallel_spacing_violations(board, rules);
    let (high_speed_adjacent_layer_parallel_violations, high_speed_adjacent_layer_parallel_pts) =
        detect_high_speed_adjacent_layer_parallel_violations(board, rules);
    let (reference_plane_margin_violations, reference_plane_margin_pts) =
        detect_reference_plane_margin_violations(board, rules);
    let (reference_plane_absence_violations, reference_plane_absence_pts) =
        detect_reference_plane_absence_violations(board);
    let (inner_layer_dual_ground_reference_violations, inner_layer_dual_ground_reference_pts) =
        detect_inner_layer_dual_ground_reference_violations(board);
    let (power_reference_stitching_cap_violations, power_reference_stitching_cap_pts) =
        detect_power_reference_stitching_cap_violations(board, comps, rules);
    let (reference_plane_intrusion_violations, reference_plane_intrusion_pts) =
        detect_reference_plane_intrusion_violations(board);
    let (ground_plane_fragmentation_violations, ground_plane_fragmentation_pts) =
        detect_ground_plane_fragmentation_violations(board);
    let (split_domain_reference_violations, split_domain_reference_pts) =
        detect_split_domain_reference_violations(board);
    let (mixed_domain_shared_reference_violations, mixed_domain_shared_reference_pts) =
        detect_mixed_domain_shared_reference_violations(board, rules);
    let (virtual_split_crossing_violations, virtual_split_crossing_pts) =
        detect_virtual_split_crossing_violations(board);
    let (high_speed_stub_violations, high_speed_stub_pts) =
        detect_high_speed_stub_violations(board);
    let (high_speed_transition_ground_via_violations, high_speed_transition_ground_via_pts) =
        detect_high_speed_transition_ground_via_violations(board, rules);
    let (
        diff_pair_transition_ground_via_symmetry_violations,
        diff_pair_transition_ground_via_symmetry_pts,
    ) = detect_diff_pair_transition_ground_via_symmetry_violations(board, rules);
    let (high_speed_terminal_ground_via_violations, high_speed_terminal_ground_via_pts) =
        detect_high_speed_terminal_ground_via_violations(board, rules);
    let (high_speed_via_pad_proximity_violations, high_speed_via_pad_proximity_pts) =
        detect_high_speed_via_pad_proximity_violations(board, rules);
    let (high_speed_via_diameter_violations, high_speed_via_diameter_pts) =
        detect_high_speed_via_diameter_violations(board, rules);
    let (blind_buried_via_drill_violations, blind_buried_via_drill_pts) =
        detect_blind_buried_via_drill_violations(board, rules);
    let (microvia_aspect_violations, microvia_aspect_violations_pts) =
        detect_microvia_aspect_violations(board, rules, build_up_mm);
    let (decoupling_ground_via_violations, decoupling_ground_via_pts) =
        detect_decoupling_ground_via_violations(board, comps, lib, rules);
    let (decoupling_power_layer_violations, decoupling_power_layer_pts) =
        detect_decoupling_power_layer_violations(board, comps, lib);
    let (decoupling_loop_area_violations, decoupling_loop_area_pts) =
        detect_decoupling_loop_area_violations(comps, lib, rules);
    let (active_ic_power_plane_violations, active_ic_power_plane_pts) =
        detect_active_ic_power_plane_violations(board, comps, lib);
    let (charge_reservoir_violations, charge_reservoir_violations_pts) =
        detect_charge_reservoir_violations(comps, lib, rules);
    hotspots.extend(charge_reservoir_violations_pts);
    let (surge_suppressor_via_violations, surge_suppressor_via_pts) =
        detect_surge_suppressor_via_violations(board, comps, lib);
    let (high_speed_via_stub_violations, high_speed_via_stub_pts) =
        detect_high_speed_via_stub_violations(board, rules);
    let (unfilled_via_in_pad_violations, unfilled_via_in_pad_pts) =
        detect_unfilled_via_in_pad_violations(board);
    let (split_plane_crossings, split_plane_pts) =
        detect_split_plane_crossings(board, comps, rules);

    hotspots.extend(sharp_pts);
    hotspots.extend(track_crossing_pts);
    hotspots.extend(hole_clearance_pts);
    hotspots.extend(serpentine_pts);
    hotspots.extend(via_spacing_pts);
    hotspots.extend(plane_hotspot_via_spacing_pts);
    hotspots.extend(diff_pair_pts);
    hotspots.extend(diff_pair_interface_mismatch_pts);
    hotspots.extend(diff_pair_coupling_cap_symmetry_pts);
    hotspots.extend(diff_pair_coupling_cap_package_pts);
    hotspots.extend(diff_pair_stitching_cap_symmetry_pts);
    hotspots.extend(diff_pair_pad_entry_mismatch_pts);
    hotspots.extend(diff_pair_pad_entry_length_pts);
    hotspots.extend(parallel_bus_length_mismatch_pts);
    hotspots.extend(diff_pair_keepout_pts);
    hotspots.extend(high_speed_edge_pts);
    hotspots.extend(high_speed_component_edge_pts);
    hotspots.extend(high_speed_termination_placement_pts);
    hotspots.extend(high_speed_parallel_spacing_pts);
    hotspots.extend(high_speed_adjacent_layer_parallel_pts);
    hotspots.extend(reference_plane_margin_pts);
    hotspots.extend(reference_plane_absence_pts);
    hotspots.extend(inner_layer_dual_ground_reference_pts);
    hotspots.extend(power_reference_stitching_cap_pts);
    hotspots.extend(reference_plane_intrusion_pts);
    hotspots.extend(ground_plane_fragmentation_pts);
    hotspots.extend(split_domain_reference_pts);
    hotspots.extend(mixed_domain_shared_reference_pts);
    hotspots.extend(virtual_split_crossing_pts);
    hotspots.extend(high_speed_stub_pts);
    hotspots.extend(high_speed_transition_ground_via_pts);
    hotspots.extend(diff_pair_transition_ground_via_symmetry_pts);
    hotspots.extend(high_speed_terminal_ground_via_pts);
    hotspots.extend(high_speed_via_pad_proximity_pts);
    hotspots.extend(high_speed_via_diameter_pts);
    hotspots.extend(blind_buried_via_drill_pts);
    hotspots.extend(microvia_aspect_violations_pts);
    hotspots.extend(decoupling_ground_via_pts);
    hotspots.extend(decoupling_power_layer_pts);
    hotspots.extend(decoupling_loop_area_pts);
    hotspots.extend(active_ic_power_plane_pts);
    hotspots.extend(surge_suppressor_via_pts);
    hotspots.extend(high_speed_via_stub_pts);
    hotspots.extend(unfilled_via_in_pad_pts);
    hotspots.extend(split_plane_pts);

    let isolation = schematic_isolation_bfs(board, comps, lib);
    let ac_coupling = parasitic_ac_coupling_check(board);

    let net_points = |net_name: &str| -> Vec<Point> {
        let mut pts = Vec::new();
        if let Some(net_id) = board.net_by_name(net_name) {
            for pad in board.pads_of(net_id) {
                pts.push(pad.pos);
            }
            for track in &board.tracks {
                if track.net == net_id {
                    let mid = Point::new(
                        Nm((track.start.x.0 + track.end.x.0) / 2),
                        Nm((track.start.y.0 + track.end.y.0) / 2),
                    );
                    pts.push(mid);
                }
            }
            for via in &board.vias {
                if via.net == net_id {
                    pts.push(via.pos);
                }
            }
        }
        pts
    };

    let comp_point = |refdes: &str| -> Option<Point> {
        comps
            .iter()
            .find(|c| c.refdes == refdes)
            .map(|c| c.placement.pos)
    };

    for violation in &isolation.violations {
        for (idx, name) in violation.path.iter().enumerate() {
            if idx % 2 == 0 {
                hotspots.extend(net_points(name));
            } else if let Some(pt) = comp_point(name) {
                hotspots.push(pt);
            }
        }
    }

    hotspots.extend(ac_coupling.hotspots.clone());

    let isolation_violations = isolation.violations.len();
    let ac_coupling_violations = ac_coupling.violations.len();

    let (charge_recycling_violations, charge_recycling_violations_pts) =
        detect_charge_recycling_violations_board(board, comps, lib);
    hotspots.extend(charge_recycling_violations_pts);
    let (pulse_skip_violations, _) = detect_pulse_skip_violations(board, rules);

    FaultReport {
        crossings,
        clearance_violations: clearance,
        near_shorts: near,
        crosstalk: crosstalk_n,
        via_adjacency: via_adj,
        acid_traps: traps,
        via_count,
        dangling,
        isolation_violations,
        ac_coupling_violations,
        sharp_bends,
        track_crossing_violations,
        hole_clearance_violations,
        serpentine_spacing_violations: serpentine_spacing,
        serpentine_length_violations: serpentine_length,
        serpentine_compensation_distance_violations: serpentine_compensation_distance,
        via_spacing_violations: via_spacing,
        plane_hotspot_via_spacing_violations: plane_hotspot_via_spacing,
        diff_pair_violations,
        diff_pair_layer_mismatch_violations,
        diff_pair_interface_layer_mismatch_violations,
        diff_pair_interface_via_count_mismatch_violations,
        diff_pair_via_count_violations,
        diff_pair_length_mismatch_violations,
        diff_pair_segment_length_mismatch_violations,
        diff_pair_spacing_variation_violations,
        diff_pair_via_symmetry_violations,
        diff_pair_coupling_cap_symmetry_violations,
        diff_pair_coupling_cap_package_violations,
        diff_pair_stitching_cap_symmetry_violations,
        diff_pair_pad_entry_mismatch_violations,
        diff_pair_pad_entry_length_violations,
        parallel_bus_length_mismatch_violations,
        diff_pair_total_length_mismatch_mm,
        diff_pair_keepout_violations,
        high_speed_edge_violations,
        high_speed_component_edge_violations,
        high_speed_termination_placement_violations,
        high_speed_parallel_spacing_violations,
        high_speed_adjacent_layer_parallel_violations,
        reference_plane_margin_violations,
        reference_plane_absence_violations,
        inner_layer_dual_ground_reference_violations,
        power_reference_stitching_cap_violations,
        reference_plane_intrusion_violations,
        ground_plane_fragmentation_violations,
        split_domain_reference_violations,
        mixed_domain_shared_reference_violations,
        virtual_split_crossing_violations,
        high_speed_stub_violations,
        high_speed_transition_ground_via_violations,
        diff_pair_transition_ground_via_symmetry_violations,
        high_speed_terminal_ground_via_violations,
        high_speed_via_pad_proximity_violations,
        high_speed_via_diameter_violations,
        blind_buried_via_drill_violations,
        microvia_aspect_violations,
        decoupling_ground_via_violations,
        decoupling_power_layer_violations,
        decoupling_loop_area_violations,
        active_ic_power_plane_violations,
        charge_reservoir_violations,
        surge_suppressor_via_violations,
        high_speed_via_stub_violations,
        unfilled_via_in_pad_violations,
        split_plane_crossings,
        charge_recycling_violations,
        pulse_skip_violations,
        // Weights reflect fab severity: a clearance violation, dangling end, or acid trap is a hard
        // reject; via-adjacency overlaps annular rings; near-shorts/crossings are graded margins;
        // via_count is a mild cost term (drill defect probability proportional to count).
        risk_score: crossings as f64
            + clearance as f64 * 20.0
            + near_score
            + dangling as f64 * 2.0
            + crosstalk_n as f64 * 0.2
            + via_adj as f64 * 5.0
            + traps as f64 * 3.0
            + via_count as f64 * 0.05
            + isolation_violations as f64 * 50.0
            + ac_coupling_violations as f64 * 10.0
            + sharp_bends as f64 * 3.0
            + track_crossing_violations as f64 * 25.0
            + hole_clearance_violations as f64 * 20.0
            + serpentine_spacing as f64 * 5.0
            + serpentine_length as f64 * 3.0
            + serpentine_compensation_distance as f64 * 5.0
            + via_spacing as f64 * 10.0
            + plane_hotspot_via_spacing as f64 * 10.0
            + diff_pair_violations as f64 * 15.0
            + diff_pair_layer_mismatch_violations as f64 * 15.0
            + diff_pair_interface_layer_mismatch_violations as f64 * 12.0
            + diff_pair_interface_via_count_mismatch_violations as f64 * 12.0
            + diff_pair_via_count_violations as f64 * 12.0
            + diff_pair_length_mismatch_violations as f64 * 12.0
            + diff_pair_segment_length_mismatch_violations as f64 * 12.0
            + diff_pair_spacing_variation_violations as f64 * 12.0
            + diff_pair_via_symmetry_violations as f64 * 12.0
            + diff_pair_coupling_cap_symmetry_violations as f64 * 12.0
            + diff_pair_coupling_cap_package_violations as f64 * 12.0
            + diff_pair_stitching_cap_symmetry_violations as f64 * 12.0
            + diff_pair_pad_entry_mismatch_violations as f64 * 12.0
            + diff_pair_pad_entry_length_violations as f64 * 12.0
            + parallel_bus_length_mismatch_violations as f64 * 12.0
            + diff_pair_total_length_mismatch_mm * 60.0
            + diff_pair_keepout_violations as f64 * 12.0
            + high_speed_edge_violations as f64 * 10.0
            + high_speed_component_edge_violations as f64 * 10.0
            + high_speed_termination_placement_violations as f64 * 12.0
            + high_speed_parallel_spacing_violations as f64 * 12.0
            + high_speed_adjacent_layer_parallel_violations as f64 * 12.0
            + reference_plane_margin_violations as f64 * 20.0
            + reference_plane_absence_violations as f64 * 25.0
            + inner_layer_dual_ground_reference_violations as f64 * 25.0
            + power_reference_stitching_cap_violations as f64 * 20.0
            + reference_plane_intrusion_violations as f64 * 20.0
            + ground_plane_fragmentation_violations as f64 * 20.0
            + split_domain_reference_violations as f64 * 20.0
            + mixed_domain_shared_reference_violations as f64 * 20.0
            + virtual_split_crossing_violations as f64 * 20.0
            + high_speed_stub_violations as f64 * 12.0
            + high_speed_transition_ground_via_violations as f64 * 15.0
            + diff_pair_transition_ground_via_symmetry_violations as f64 * 15.0
            + high_speed_terminal_ground_via_violations as f64 * 15.0
            + high_speed_via_pad_proximity_violations as f64 * 12.0
            + high_speed_via_diameter_violations as f64 * 12.0
            + blind_buried_via_drill_violations as f64 * 12.0
            + microvia_aspect_violations as f64 * 12.0
            + decoupling_ground_via_violations as f64 * 12.0
            + decoupling_power_layer_violations as f64 * 12.0
            + decoupling_loop_area_violations as f64 * 12.0
            + active_ic_power_plane_violations as f64 * 12.0
            + charge_reservoir_violations as f64 * 20.0
            + surge_suppressor_via_violations as f64 * 12.0
            + high_speed_via_stub_violations as f64 * 15.0
            + unfilled_via_in_pad_violations as f64 * 15.0
            + split_plane_crossings as f64 * 25.0
            + charge_recycling_violations as f64 * 10.0
            + pulse_skip_violations as f64 * 8.0,
        hotspots,
    }
}
