use super::*;
use crate::board::{Board, LayerId, NetClassKind, NetId, Pad, Track, Via, ViaKind, Zone, ZoneFill};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::{Component, FootprintDef, Role};
use crate::audit::detect_power::{detect_charge_recycling_violations_board, detect_pulse_skip_violations};
use crate::rules::DesignRules;

mod diff_pair;
mod high_speed;
mod pdn;
mod routing;
mod via;
mod integration;

pub(super) fn board() -> Board {
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    Board::new(spec)
}

#[test]
fn hard_drc_clean_covers_optimizer_rejection_fields() {
    let clean = FaultReport::default();
    assert!(
        clean.hard_drc_clean(),
        "a default report has no hard DRC faults"
    );

    type FaultMarker = fn(&mut FaultReport);
    let dirty_fields: [(&str, FaultMarker); 66] = [
        ("clearance violations", |r: &mut FaultReport| {
            r.clearance_violations = 1;
        }),
        ("via adjacency", |r: &mut FaultReport| {
            r.via_adjacency = 1;
        }),
        ("acid traps", |r: &mut FaultReport| {
            r.acid_traps = 1;
        }),
        ("dangling", |r: &mut FaultReport| r.dangling = 1),
        ("sharp bend", |r: &mut FaultReport| r.sharp_bends = 1),
        ("track crossing", |r: &mut FaultReport| {
            r.track_crossing_violations = 1;
        }),
        ("hole clearance", |r: &mut FaultReport| {
            r.hole_clearance_violations = 1;
        }),
        ("serpentine spacing", |r: &mut FaultReport| {
            r.serpentine_spacing_violations = 1;
        }),
        ("serpentine length", |r: &mut FaultReport| {
            r.serpentine_length_violations = 1;
        }),
        ("serpentine compensation distance", |r: &mut FaultReport| {
            r.serpentine_compensation_distance_violations = 1;
        }),
        ("different-net via spacing", |r: &mut FaultReport| {
            r.via_spacing_violations = 1;
        }),
        ("plane hotspot via spacing", |r: &mut FaultReport| {
            r.plane_hotspot_via_spacing_violations = 1;
        }),
        ("diff-pair violation", |r: &mut FaultReport| {
            r.diff_pair_violations = 1;
        }),
        ("diff-pair layer mismatch", |r: &mut FaultReport| {
            r.diff_pair_layer_mismatch_violations = 1;
        }),
        (
            "diff-pair interface layer mismatch",
            |r: &mut FaultReport| {
                r.diff_pair_interface_layer_mismatch_violations = 1;
            },
        ),
        (
            "diff-pair interface via count mismatch",
            |r: &mut FaultReport| {
                r.diff_pair_interface_via_count_mismatch_violations = 1;
            },
        ),
        ("diff-pair via count", |r: &mut FaultReport| {
            r.diff_pair_via_count_violations = 1;
        }),
        ("diff-pair length mismatch", |r: &mut FaultReport| {
            r.diff_pair_length_mismatch_violations = 1;
        }),
        (
            "diff-pair segment length mismatch",
            |r: &mut FaultReport| {
                r.diff_pair_segment_length_mismatch_violations = 1;
            },
        ),
        ("parallel bus length mismatch", |r: &mut FaultReport| {
            r.parallel_bus_length_mismatch_violations = 1;
        }),
        ("diff-pair spacing variation", |r: &mut FaultReport| {
            r.diff_pair_spacing_variation_violations = 1;
        }),
        ("diff-pair via symmetry", |r: &mut FaultReport| {
            r.diff_pair_via_symmetry_violations = 1;
        }),
        ("diff-pair coupling cap symmetry", |r: &mut FaultReport| {
            r.diff_pair_coupling_cap_symmetry_violations = 1;
        }),
        ("diff-pair coupling cap package", |r: &mut FaultReport| {
            r.diff_pair_coupling_cap_package_violations = 1;
        }),
        ("diff-pair stitching cap symmetry", |r: &mut FaultReport| {
            r.diff_pair_stitching_cap_symmetry_violations = 1;
        }),
        ("diff-pair pad entry mismatch", |r: &mut FaultReport| {
            r.diff_pair_pad_entry_mismatch_violations = 1;
        }),
        ("diff-pair pad entry length", |r: &mut FaultReport| {
            r.diff_pair_pad_entry_length_violations = 1;
        }),
        ("diff-pair keepout", |r: &mut FaultReport| {
            r.diff_pair_keepout_violations = 1;
        }),
        ("high-speed edge", |r: &mut FaultReport| {
            r.high_speed_edge_violations = 1;
        }),
        ("high-speed component edge", |r: &mut FaultReport| {
            r.high_speed_component_edge_violations = 1;
        }),
        ("high-speed termination placement", |r: &mut FaultReport| {
            r.high_speed_termination_placement_violations = 1;
        }),
        ("high-speed parallel spacing", |r: &mut FaultReport| {
            r.high_speed_parallel_spacing_violations = 1;
        }),
        (
            "high-speed adjacent-layer parallel",
            |r: &mut FaultReport| {
                r.high_speed_adjacent_layer_parallel_violations = 1;
            },
        ),
        ("reference plane margin", |r: &mut FaultReport| {
            r.reference_plane_margin_violations = 1;
        }),
        ("reference plane absence", |r: &mut FaultReport| {
            r.reference_plane_absence_violations = 1;
        }),
        (
            "inner layer dual ground reference",
            |r: &mut FaultReport| {
                r.inner_layer_dual_ground_reference_violations = 1;
            },
        ),
        ("power reference stitching cap", |r: &mut FaultReport| {
            r.power_reference_stitching_cap_violations = 1;
        }),
        ("reference plane intrusion", |r: &mut FaultReport| {
            r.reference_plane_intrusion_violations = 1;
        }),
        ("ground plane fragmentation", |r: &mut FaultReport| {
            r.ground_plane_fragmentation_violations = 1;
        }),
        ("split-domain reference", |r: &mut FaultReport| {
            r.split_domain_reference_violations = 1;
        }),
        ("mixed-domain shared reference", |r: &mut FaultReport| {
            r.mixed_domain_shared_reference_violations = 1;
        }),
        ("virtual split crossing", |r: &mut FaultReport| {
            r.virtual_split_crossing_violations = 1;
        }),
        ("high-speed stub", |r: &mut FaultReport| {
            r.high_speed_stub_violations = 1;
        }),
        ("high-speed transition return", |r: &mut FaultReport| {
            r.high_speed_transition_ground_via_violations = 1;
        }),
        (
            "diff-pair transition return symmetry",
            |r: &mut FaultReport| {
                r.diff_pair_transition_ground_via_symmetry_violations = 1;
            },
        ),
        ("high-speed terminal ground via", |r: &mut FaultReport| {
            r.high_speed_terminal_ground_via_violations = 1;
        }),
        ("high-speed via pad proximity", |r: &mut FaultReport| {
            r.high_speed_via_pad_proximity_violations = 1;
        }),
        ("high-speed via diameter", |r: &mut FaultReport| {
            r.high_speed_via_diameter_violations = 1;
        }),
        ("blind/buried via drill", |r: &mut FaultReport| {
            r.blind_buried_via_drill_violations = 1;
        }),
        ("microvia aspect", |r: &mut FaultReport| {
            r.microvia_aspect_violations = 1;
        }),
        ("decoupling ground via", |r: &mut FaultReport| {
            r.decoupling_ground_via_violations = 1;
        }),
        ("decoupling power layer", |r: &mut FaultReport| {
            r.decoupling_power_layer_violations = 1;
        }),
        ("decoupling loop area", |r: &mut FaultReport| {
            r.decoupling_loop_area_violations = 1;
        }),
        ("active IC power plane", |r: &mut FaultReport| {
            r.active_ic_power_plane_violations = 1;
        }),
        ("charge reservoir", |r: &mut FaultReport| {
            r.charge_reservoir_violations = 1;
        }),
        ("high-speed via stub", |r: &mut FaultReport| {
            r.high_speed_via_stub_violations = 1;
        }),
        ("unfilled via in pad", |r: &mut FaultReport| {
            r.unfilled_via_in_pad_violations = 1;
        }),
        ("surge suppressor via", |r: &mut FaultReport| {
            r.surge_suppressor_via_violations = 1;
        }),
        ("split-plane crossing", |r: &mut FaultReport| {
            r.split_plane_crossings = 1;
        }),
        ("transmission line length", |r: &mut FaultReport| {
            r.transmission_line_length_violations = 1;
        }),
        ("decoupling cap distance", |r: &mut FaultReport| {
            r.decoupling_cap_distance_violations = 1;
        }),
        ("antenna impedance", |r: &mut FaultReport| {
            r.antenna_impedance_violations = 1;
        }),
        ("ground via stitching", |r: &mut FaultReport| {
            r.ground_via_stitching_violations = 1;
        }),
        ("via return count", |r: &mut FaultReport| {
            r.high_speed_via_return_count_violations = 1;
        }),
        ("cap dielectric grade", |r: &mut FaultReport| {
            r.cap_dielectric_grade_violations = 1;
        }),
        ("through-hole high-speed", |r: &mut FaultReport| {
            r.through_hole_high_speed_violations = 1;
        }),
    ];

    for (name, mark_dirty) in dirty_fields {
        let mut report = FaultReport::default();
        mark_dirty(&mut report);
        assert!(
            !report.hard_drc_clean(),
            "{name} must reject optimizer clean-board selection"
        );
    }
}

/// Meta-test: every `self.X == 0` clause inside `hard_drc_clean` (the optimizer-rejection
/// gate) must have a matching `r.X = N` set inside the `dirty_fields` array of
/// `hard_drc_clean_covers_optimizer_rejection_fields`. Parses `audit.rs` at build time via
/// `include_str!` so adding a clause to `hard_drc_clean` -- or removing one from
/// `dirty_fields` -- fails this assertion even if every other test passes.
///
/// Hand-rolled prefix scans rather than regex to keep the test robust against audit.rs
/// formatting drift. The two scans are intentionally tolerant: any line of hard_drc_clean's
/// body beginning with `&& self.` or `self.` followed by an identifier and ` == 0`; any
/// line inside the dirty_fields array containing `r.` followed by an identifier and `=`. The
/// failure message names every missing field so the maintainer can extend dirty_fields in
/// one shot.
#[test]
fn dirty_fields_mirrors_every_hard_drc_clean_clause() {
    use std::collections::BTreeSet;
    const SRC: &str = include_str!("../fault_report.rs");
    // Post-carve, the dirty_fields array lives in tests/mod.rs itself — meta-test scans its own
    // source so a future file move doesn't silently bypass the gate.
    const SRC_DF: &str = include_str!("mod.rs");

    // Step 1: every `self.X == 0` clause inside the hard_drc_clean body.
    let lines: Vec<&str> = SRC.lines().collect();
    let mut hard_fields = BTreeSet::new();
    let mut in_hard_drc = false;
    let mut brace_depth: i32 = 0;
    for line in &lines {
        let t = line.trim();
        if !in_hard_drc {
            if t.starts_with("pub(crate) fn hard_drc_clean(")
                || t.starts_with("pub fn hard_drc_clean(")
            {
                in_hard_drc = true;
                brace_depth += t.matches('{').count() as i32 - t.matches('}').count() as i32;
            }
            continue;
        }
        brace_depth += t.matches('{').count() as i32 - t.matches('}').count() as i32;
        if brace_depth <= 0 {
            break;
        }
        if brace_depth != 1 {
            continue; // ignore nested-closure clauses inside hard_drc_clean
        }
        if t.starts_with("//") || t.starts_with("/*") {
            continue; // ignore Rust comments (line or block) inside hard_drc_clean body
        }
        if !t.contains(" == 0") {
            continue;
        }
        // First clause: `self.X == 0`. Continuation: `&& self.X == 0`.
        let rest = if let Some(r) = t.strip_prefix("&& self.") {
            r
        } else if let Some(r) = t.strip_prefix("self.") {
            r
        } else {
            continue;
        };
        let end = rest
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        let field = rest[..end].trim_end_matches([' ', '\t']);
        if !field.is_empty() {
            hard_fields.insert(field.to_string());
        }
    }

    // Step 2: every `r.X = N` set inside the dirty_fields array body. Post-carve the array
    // lives in `tests/mod.rs` (this file), so we re-read SRC_DF instead of `fault_report.rs`.
    let df_lines: Vec<&str> = SRC_DF.lines().collect();
    let start = df_lines
        .iter()
        .position(|l| {
            let t = l.trim_start();
            (t.starts_with("let ") || t.starts_with("let mut "))
                && t.contains("dirty_fields:")
                && t.contains("= [")
        })
        .expect("dirty_fields array exists");
    let end = df_lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find(|(_, l)| l.trim() == "];")
        .map(|(i, _)| i)
        .expect("dirty_fields terminates with ];");
    let array_text = df_lines[start..=end + 1].join("\n");

    let mut dirty_field_sets = BTreeSet::new();
    for line in array_text.lines() {
        let t_line = line.trim_start();
        if t_line.starts_with("//") {
            continue;
        }
        // Find the first assignment target substring. This handles one-line closures such as
        // `("dangling", |r| r.dangling = 1)` and multi-line bodies with `r.X = 1;`.
        let Some(rel_idx) = line.find("r.") else {
            continue;
        };
        let after_r = &line[rel_idx + 2..];
        let Some(eq_idx) = after_r.find('=') else {
            continue;
        };
        let before = after_r[..eq_idx].trim_end_matches([' ', '\t']);
        let end = before
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(before.len());
        let field = before[..end].trim_end_matches([' ', '\t']);
        if !field.is_empty() {
            dirty_field_sets.insert(field.to_string());
        }
    }

    // Step 3: diff and report any missing entries.
    let missing: Vec<String> = hard_fields
        .iter()
        .filter(|f| !dirty_field_sets.contains(*f))
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "every `self.X == 0` clause in hard_drc_clean must have a matching `r.X = N` set              inside dirty_fields (this file, near `hard_drc_clean_covers_optimizer_rejection_fields`)              so adding a clause to hard_drc_clean cannot silently bypass the optimizer-rejection              test. Missing entries: {missing:?}. Extend dirty_fields in that test to add one entry              per missing field, setting that field to 1."
    );
}
