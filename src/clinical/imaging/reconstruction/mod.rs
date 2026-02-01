//! Clinical Image Reconstruction Workflows
//!
//! This module provides production-ready reconstruction pipelines for clinical
//! ultrasound imaging applications, including real-time SIRT reconstruction
//! with streaming data support and quality monitoring.

pub mod clinical_monitoring;
pub mod real_time_sirt;

pub use clinical_monitoring::{
    ClinicalMonitor, MonitoringConfig, MonitoringReport, SafetyEvent, SafetyEventType,
    SafetySeverity,
};
pub use real_time_sirt::{
    FrameQuality, RealTimeSirtConfig, RealTimeSirtPipeline, ReconstructionFrame,
};
