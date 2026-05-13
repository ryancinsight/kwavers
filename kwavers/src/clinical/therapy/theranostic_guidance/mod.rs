//! Clinical same-device ultrasound treatment and FWI/RTM monitoring workflow.
//!
//! This module owns patient/anatomy selection, CT-derived placement context,
//! treatment-device analog geometry, pressure-calibrated exposure synthesis,
//! and clinical reconstruction metrics for the book figures. It is not a
//! proprietary device model. Generic seismic FWI and RTM kernels remain in
//! `crate::solver::inverse::seismic`; this clinical workflow composes the
//! same-device finite-frequency monitoring model used for INSIGHTEC-like
//! transcranial and HistoSonics-like abdominal scenarios.

mod aperture;
mod config;
mod context;
mod exposure;
mod geometry;
mod helmet3d;
mod medium;
mod metrics;
mod skin;
mod solver;

pub use config::{AnatomyKind, TheranosticFwiConfig};
pub use context::{
    build_abdominal_placement_context, build_brain_placement_context, PlacementContext,
    Point3 as PlacementPoint3,
};
pub use geometry::{placement_metrics, DeviceLayout, DevicePlacementMetrics, Point2};
pub use helmet3d::{plan_brain_helmet_placement, BrainHelmetPlacement3D, Point3};
pub use medium::{prepare_abdominal_slice, prepare_brain_slice, PreparedTheranosticSlice};
pub use metrics::ReconstructionMetrics;
pub use solver::{
    run_theranostic_fwi, TheranosticFwiResult, THERANOSTIC_OPERATOR_BACKEND,
    THERANOSTIC_OPERATOR_MODEL,
};

#[cfg(test)]
mod tests;
