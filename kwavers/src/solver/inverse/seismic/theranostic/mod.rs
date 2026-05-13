//! Same-device ultrasound treatment and FWI/RTM monitoring simulations.
//!
//! This module implements the model-consistent theranostic slice used by the
//! book figures. It is not a proprietary device model. It exposes public
//! geometry analogs for an INSIGHTEC-like head helmet and a HistoSonics-like
//! abdominal histotripsy head, then simulates exposure, active finite-frequency
//! FWI, passive subharmonic source inversion, nonlinear harmonic inversion, and
//! fusion from the same element layout.

mod aperture;
mod config;
mod context;
mod geometry;
mod helmet3d;
mod medium;
mod metrics;
mod operator;
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
pub use solver::{run_theranostic_fwi, TheranosticFwiResult};

#[cfg(test)]
mod tests;
