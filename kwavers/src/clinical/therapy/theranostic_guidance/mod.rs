//! Clinical same-device ultrasound treatment and finite-frequency inverse/RTM monitoring workflow.
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
mod misfit;
mod nonlinear3d;
mod skin;
mod solver;
pub mod standing_wave_opt;
mod waveform;

pub use config::{AnatomyKind, TheranosticInverseConfig};
pub use context::{
    build_abdominal_placement_context, build_brain_placement_context, PlacementContext,
    Point3 as PlacementPoint3,
};
pub use geometry::{placement_metrics, DeviceLayout, DevicePlacementMetrics, Point2};
pub use helmet3d::{plan_brain_helmet_placement, BrainHelmetPlacement3D, Point3};
pub use medium::{prepare_abdominal_slice, prepare_brain_slice, PreparedTheranosticSlice};
pub use metrics::ReconstructionMetrics;
pub use misfit::WaveformMisfit;
pub use nonlinear3d::{
    run_theranostic_nonlinear_3d, Nonlinear3dConfig, Nonlinear3dResult,
    VolumeReconstructionMetrics, THERANOSTIC_CAVITATION_INVERSE_MODEL,
    THERANOSTIC_NONLINEAR_3D_MODEL, THERANOSTIC_NONLINEAR_3D_PROPAGATOR,
};
pub use solver::{
    run_theranostic_inverse, TheranosticInverseResult, THERANOSTIC_FULL_WAVE_INVERSION,
    THERANOSTIC_INVERSE_MODEL_FAMILY, THERANOSTIC_NONLINEAR_WAVE_PROPAGATION,
    THERANOSTIC_OPERATOR_BACKEND, THERANOSTIC_OPERATOR_MODEL,
};
pub use standing_wave_opt::{run_standing_wave_suppression, StandingWaveOptConfig, StandingWaveOptResult};
pub use waveform::{
    simulate_waveform_adjoint_rtm, WaveformSimulationResult, THERANOSTIC_WAVEFORM_MODEL,
};

#[cfg(test)]
mod tests;
