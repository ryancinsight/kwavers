//! Calibration procedures for flexible array geometries.
//!
//! Provides methods for calibrating and tracking flexible transducer arrays,
//! including self-calibration and external tracking integration.

pub mod manager;
pub mod types;

pub use manager::CalibrationManager;
pub use types::{CalibrationData, GeometrySnapshot, QualityMetrics};
