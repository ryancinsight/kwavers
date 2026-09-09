//! Board-level DFM / physics critic.
//!
//! The single public entry point is [`audit`], which calls all per-family detectors
//! and accumulates results into a [`FaultReport`] with a weighted risk score.
//! Hotspot rasterisation and efficiency-audit helpers for the placement feedback
//! loop are also exposed here.

mod audit;
pub mod congestion;
pub mod diagnostics;

pub use audit::audit;
pub use congestion::{rasterize_hotspots, rasterize_hotspots_radius, weakness_field};
pub use diagnostics::{
    charge_recycling_efficiency_audit, pulse_skip_interference_audit, ChargeRecyclingReport,
    PulseSkipInterferenceReport,
};
