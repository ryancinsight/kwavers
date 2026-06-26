//! High-speed routing violation detectors.
//!
//! Covers board-edge clearance, component edge clearance, termination placement,
//! parallel spacing (same and adjacent layer), reference-plane margin/absence/
//! intrusion/stitching, split-domain crossings, via stub/transition/terminal,
//! via diameter, and blind/buried via drill size.

mod edge_spacing;
mod via_termination;
mod reference_plane;
mod split_domain;

pub(crate) use edge_spacing::{
    detect_high_speed_edge_violations,
    detect_high_speed_component_edge_violations,
    detect_high_speed_parallel_spacing_violations,
    detect_high_speed_adjacent_layer_parallel_violations,
};

pub(crate) use via_termination::{
    detect_high_speed_termination_placement_violations,
    detect_high_speed_stub_violations,
    detect_high_speed_transition_ground_via_violations,
    detect_high_speed_terminal_ground_via_violations,
    detect_high_speed_via_pad_proximity_violations,
    detect_high_speed_via_diameter_violations,
    detect_blind_buried_via_drill_violations,
};

pub(crate) use reference_plane::{
    detect_reference_plane_margin_violations,
    detect_reference_plane_absence_violations,
    detect_inner_layer_dual_ground_reference_violations,
    detect_power_reference_stitching_cap_violations,
    detect_reference_plane_intrusion_violations,
    detect_ground_plane_fragmentation_violations,
};

pub(crate) use split_domain::{
    detect_split_domain_reference_violations,
    detect_mixed_domain_shared_reference_violations,
    detect_virtual_split_crossing_violations,
};
