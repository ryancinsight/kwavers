//! Differential-pair violation detectors.
//!
//! Covers layer/via-count symmetry, length matching, spacing uniformity, pad
//! entry, coupling caps, interface consistency, keepout, stitching-cap symmetry,
//! and transition-via symmetry.

mod cap;
mod interface;
mod routing;

pub(crate) use routing::{
    detect_diff_pair_keepout_violations, detect_diff_pair_pad_entry_length_violations,
    detect_diff_pair_pad_entry_mismatch_violations, detect_diff_pair_violations, diff_pair_axis,
};

pub(crate) use cap::{
    detect_diff_pair_coupling_cap_package_violations,
    detect_diff_pair_coupling_cap_symmetry_violations,
    detect_diff_pair_stitching_cap_symmetry_violations,
};

pub(crate) use interface::{
    detect_diff_pair_interface_mismatch_violations,
    detect_diff_pair_transition_ground_via_symmetry_violations,
};

/// Propagation axis of a diff pair (the axis along which the two members run side-by-side).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairAxis {
    X,
    Y,
}
