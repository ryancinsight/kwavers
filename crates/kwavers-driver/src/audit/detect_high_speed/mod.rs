//! High-speed routing violation detectors.
//!
//! Covers board-edge clearance, component edge clearance, termination placement,
//! parallel spacing (same and adjacent layer), reference-plane margin/absence/
//! intrusion/stitching, split-domain crossings, via stub/transition/terminal,
//! via diameter, blind/buried via drill size, the λ/10 transmission-line length
//! rule, ground-via stitching density, signal-via return-via count, and
//! through-hole/leaded package detection.

mod edge_spacing;
mod ground_via_stitching;
mod package_form;
mod reference_plane;
mod split_domain;
mod trace_length;
mod via_termination;

pub(crate) use edge_spacing::{
    detect_high_speed_adjacent_layer_parallel_violations,
    detect_high_speed_component_edge_violations, detect_high_speed_edge_violations,
    detect_high_speed_parallel_spacing_violations,
};

pub(crate) use ground_via_stitching::{
    detect_ground_via_stitching_violations, detect_high_speed_via_return_count_violations,
};

pub(crate) use package_form::detect_through_hole_high_speed_violations;

pub(crate) use trace_length::detect_transmission_line_length_violations;

pub(crate) use via_termination::{
    detect_blind_buried_via_drill_violations, detect_high_speed_stub_violations,
    detect_high_speed_terminal_ground_via_violations,
    detect_high_speed_termination_placement_violations,
    detect_high_speed_transition_ground_via_violations, detect_high_speed_via_diameter_violations,
    detect_high_speed_via_pad_proximity_violations,
};

pub(crate) use reference_plane::{
    detect_ground_plane_fragmentation_violations,
    detect_inner_layer_dual_ground_reference_violations,
    detect_power_reference_stitching_cap_violations, detect_reference_plane_absence_violations,
    detect_reference_plane_intrusion_violations, detect_reference_plane_margin_violations,
};

pub(crate) use split_domain::{
    detect_mixed_domain_shared_reference_violations, detect_split_domain_reference_violations,
    detect_virtual_split_crossing_violations,
};
