//! Clinical same-device ultrasound treatment and finite-frequency inverse/RTM monitoring workflow.
//!
//! This module owns patient/anatomy selection, CT-derived placement context,
//! treatment-device analog geometry, pressure-calibrated exposure synthesis,
//! and clinical reconstruction metrics for the book figures. It is not a
//! proprietary device model. Generic seismic FWI and RTM kernels remain in
//! `kwavers_solver::inverse::seismic`; this clinical workflow composes the
//! same-device finite-frequency monitoring model used for transcranial and
//! abdominal focused-bowl scenarios.

mod abdominal3d;
mod aperture;
mod config;
mod context;
mod elastic_shear;
mod exposure;
mod geometry;
mod medium;
mod metrics;
mod misfit;
mod nonlinear3d;
mod scene;
mod skin;
mod solver;
pub mod standing_wave_opt;
pub mod synthetic;
mod transcranial_focused_bowl3d;
mod transcranial_fus;
mod transmit_schedule;
mod waveform;

pub use abdominal3d::{plan_abdominal_array_placement, AbdominalArrayPlacement3D};
pub use config::{AnatomyKind, PassiveReconstructionMode, TheranosticInverseConfig};
pub use context::{
    build_abdominal_placement_context, build_brain_placement_context, PlacementContext,
};
pub use elastic_shear::{ElasticShearReconstructionResult, THERANOSTIC_ELASTIC_SHEAR_MODEL};
pub use geometry::{placement_metrics, DeviceLayout, DevicePlacementMetrics, Point2, Point3};
pub use medium::{
    prepare_abdominal_slice, prepare_brain_slice, BrainTargetSelection, PreparedTheranosticSlice,
};
pub use metrics::ReconstructionMetrics;
pub use misfit::WaveformMisfit;
pub use nonlinear3d::{
    run_theranostic_nonlinear_3d, Nonlinear3dConfig, Nonlinear3dResult,
    VolumeReconstructionMetrics, THERANOSTIC_CAVITATION_INVERSE_MODEL,
    THERANOSTIC_NONLINEAR_3D_MODEL, THERANOSTIC_NONLINEAR_3D_PROPAGATOR,
};
pub use scene::{target_index_from_mask_fraction_3d, validate_target_fraction_xyz};
pub use solver::{
    run_theranostic_inverse, TheranosticInverseResult, THERANOSTIC_FULL_WAVE_INVERSION,
    THERANOSTIC_INVERSE_MODEL_FAMILY, THERANOSTIC_ITERATIVE_ELASTIC_FWI,
    THERANOSTIC_NONLINEAR_WAVE_PROPAGATION, THERANOSTIC_OPERATOR_BACKEND,
    THERANOSTIC_OPERATOR_MODEL,
};
pub use standing_wave_opt::{
    run_standing_wave_suppression, StandingWaveOptConfig, StandingWaveOptResult,
};
pub use transcranial_focused_bowl3d::{
    plan_transcranial_focused_bowl_placement, TranscranialFocusedBowlPlacement3D,
};
pub use transcranial_fus::{
    bbb_opening_dose, evaluate_pressure_field, gbm_subspot_covered_fraction, gbm_subspot_raster,
    run_skull_adaptive_transcranial_benchmark, run_transcranial_fus_planning,
    transcranial_pennes_thermal_dose, PressureFieldMetrics, SkullAdaptiveBenchmarkConfig,
    SkullAdaptiveBenchmarkResult, SkullAwareTransducerPlacement, TranscranialFusPlan,
    TranscranialFusPlanConfig, TranscranialThermalResult,
};
pub use transmit_schedule::{
    select_transmit_schedule, TransmitScheduleConfig, TransmitScheduleResult,
    TransmitScheduleStrategy, TRANSMIT_SCHEDULE_MODEL,
};
pub use waveform::{
    simulate_peak_pressure_exposure, simulate_waveform_adjoint_rtm, PeakPressureExposureResult,
    WaveformSimulationResult, THERANOSTIC_HYBRID_PSTD_FDTD_EXPOSURE_READY,
    THERANOSTIC_WAVEFORM_MODEL, THERANOSTIC_WAVE_EXPOSURE_BACKEND, THERANOSTIC_WAVE_EXPOSURE_MODEL,
};

#[cfg(test)]
mod tests;
