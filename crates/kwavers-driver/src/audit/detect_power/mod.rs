//! Power-integrity, decoupling, and via-quality auditors.
//!
//! Each `detect_*` function identifies one family of defects and returns
//! `(count, Vec<Point>)` where the points are board-space hotspot positions
//! for the congestion-weighted placement feedback loop.
//!
//! All items are `pub(crate)` — the only caller is [`crate::audit::critic::audit`].

mod decoupling;
mod plane;
mod via;

pub(crate) use decoupling::{
    detect_charge_reservoir_violations,
    detect_decoupling_loop_area_violations,
    detect_decoupling_power_layer_violations,
};

pub(crate) use plane::{
    detect_active_ic_power_plane_violations,
    detect_charge_recycling_violations_board,
    detect_pulse_skip_violations,
    detect_split_plane_crossings,
    point_projects_inside_segment,
};

pub(crate) use via::{
    detect_decoupling_ground_via_violations,
    detect_high_speed_via_stub_violations,
    detect_microvia_aspect_violations,
    detect_surge_suppressor_via_violations,
    detect_unfilled_via_in_pad_violations,
};
