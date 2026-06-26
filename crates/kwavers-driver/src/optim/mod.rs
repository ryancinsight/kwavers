//! Joint driver–thermal–acoustic co-optimization.
//!
//! This module assembles the physics models from [`crate::driver`], [`crate::physics::thermal`],
//! [`crate::physics::acoustic`], [`crate::physics::emi`], and [`crate::physics::pdn`] into a single
//! design-point evaluation. Given one operating point and board geometry, it returns a
//! [`DesignReport`] covering electrical efficiency, thermal headroom, acoustic output, EMI margin,
//! and PDN integrity — the quantities needed to decide between IC options, duty cycles, or layout
//! strategies without running individual module calls by hand.
//!
//! # Evidence tier
//!
//! All physics come from their respective modules (each carries its own evidence tier and
//! unit tests). The assembly here adds no new physics; correctness depends on correct inputs.
//!
//! # Slice layout
//!
//! Carved by **role** (Phase 4h). Plain backticks name the slice-private submodules; the public
//! items each hosts stay clickable.
//! * `context` — the input context types: [`ArrayGeometry`], [`ThermalContext`], [`PdnConfig`],
//!   [`EmiContext`].
//! * `report` — the [`DesignReport`] output aggregate.
//! * `evaluate` — [`evaluate_design_point`], the orchestrator that drives every physics module once.
//! * `kernels` — standalone design-limit helpers: [`max_safe_duty_thermal`],
//!   [`ringing_exceeds_breakdown`], [`hot_track_resistance`].

mod context;
mod evaluate;
mod kernels;
mod report;

#[cfg(test)]
mod tests;

pub use context::{ArrayGeometry, EmiContext, PdnConfig, ThermalContext};
pub use evaluate::evaluate_design_point;
pub use kernels::{hot_track_resistance, max_safe_duty_thermal, ringing_exceeds_breakdown};
pub use report::DesignReport;
