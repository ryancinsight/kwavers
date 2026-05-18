//! Real-Time Acoustic Intensity Tracking System
//!
//! Implements continuous monitoring of acoustic intensity fields
//! with temporal averaging, peak tracking, and safety-relevant metrics.
//!
//! # Metrics Provided
//!
//! - **SPTA** (Spatial Peak Temporal Average): FDA-regulated safety metric
//! - **ISPPA** (Spatial Peak Pulse Average): Peak intensity within pulse
//! - **I_tas** (Temporal Average Spatial): Time-averaged field intensity
//! - **Thermal Dose** (CEM43): Cumulative equivalent minutes at 43°C
//! - **Peak Intensity**: Maximum instantaneous pressure²/(ρc)
//!
//! # Clinical Relevance
//!
//! - SPTA < 720 mW/cm² for FDA-compliant diagnostic ultrasound
//! - SPTA > 100 W/cm² typical for therapeutic HIFU
//! - CEM43 < 240 minutes to prevent tissue necrosis
//! - Peak intensity determines cavitation nucleation
//!
//! # References
//!
//! - IEC 62359:2010 - Ultrasonics - Field characterization
//! - FDA 510(k) Guidance - Acoustic Output Measurement
//! - Sapareto & Dewey (1984) - Thermal dose determination

#[cfg(test)]
mod tests;
pub mod tracker;
pub mod types;

pub use tracker::IntensityTracker;
pub use types::{InstantaneousIntensity, IntensityTrackerDose, TemporalIntensityMetrics};
