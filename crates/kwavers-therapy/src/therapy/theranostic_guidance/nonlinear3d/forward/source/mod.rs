//! Source geometry, encoding, signal generation, and grid injection.

mod inject;
mod plan;
mod stencil;
#[cfg(test)]
mod tests;
mod travel;

pub(super) use inject::inject_sources;
pub(in crate::therapy::theranostic_guidance::nonlinear3d) use plan::source_plan_metrics;
pub(super) use plan::{build_source_plan, DriveContext};
pub(super) use travel::source_cells;
