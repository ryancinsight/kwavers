//! Clinical Image Reconstruction Workflows
//!
//! This module provides production-ready reconstruction pipelines for clinical
//! ultrasound imaging applications, including real-time SIRT reconstruction
//! with streaming data support and quality monitoring.

pub mod acoustic_projection;
pub mod breast_ust_fwi;
pub mod clinical_monitoring;
pub mod radon;
pub mod real_time_sirt;
pub mod sound_speed_shift;
pub mod transcranial_ust;

pub use acoustic_projection::AcousticProjectionGeometry;
pub use breast_ust_fwi::{
    generate_breast_ust_pstd_frequency_dataset, load_ali_2025_breast_phantom_from_hdf5,
    load_ali_2025_breast_phantom_from_hdf5_with_config, reconstruct_breast_ust_sound_speed_volume,
    BreastUstAliPhantom, BreastUstAliPhantomHdf5Config, BreastUstFwiImage,
    BreastUstPhantomStorageOrder, BreastUstPstdDataset, BreastUstPstdDatasetConfig,
    BreastUstSoundSpeedUnit, BREAST_UST_ALI_2025_PHANTOM_MODEL, BREAST_UST_FWI_MODEL,
    BREAST_UST_PSTD_DATASET_MODEL,
};
pub use clinical_monitoring::{
    ClinicalMonitor, ClinicalMonitoringConfig, MonitoringReport, MonitoringSafetyEventType,
    SafetyEvent, SafetySeverity,
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
