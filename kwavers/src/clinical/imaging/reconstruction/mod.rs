//! Clinical Image Reconstruction Workflows
//!
//! This module provides production-ready reconstruction pipelines for clinical
//! ultrasound imaging applications, including real-time SIRT reconstruction
//! with streaming data support and quality monitoring.

pub mod acoustic_projection;
pub mod clinical_monitoring;
pub mod real_time_sirt;
pub mod sound_speed_shift;

pub use acoustic_projection::AcousticProjectionGeometry;
pub use clinical_monitoring::{
    ClinicalMonitor, MonitoringConfig, MonitoringReport, SafetyEvent, SafetyEventType,
    SafetySeverity,
};
pub use real_time_sirt::{
    FrameQuality, RealTimeSirtConfig, RealTimeSirtPipeline, ReconstructionFrame,
};
pub use sound_speed_shift::{
    predict_sound_speed_time_shifts, reconstruct_sound_speed_shift, ShiftPrior, ShiftSampling,
    SoundSpeedShiftBatch, SoundSpeedShiftBatchConfig, SoundSpeedShiftBatchFrame,
    SoundSpeedShiftConfig, SoundSpeedShiftFrameSummary, SoundSpeedShiftImage,
    SoundSpeedShiftObjectiveHistoryPolicy, SoundSpeedShiftPlan, SoundSpeedShiftSample,
    SOUND_SPEED_SHIFT_MODEL,
};
