//! Flexible transducer arrays with real-time geometry tracking
//!
//! Modular implementation following SOLID principles

pub mod config;
pub mod geometry;
pub mod calibration;
pub mod array;

pub use config::{FlexibleTransducerConfig, FlexibilityModel, CalibrationMethod, TrackingSystem};
pub use geometry::{GeometryState, DeformationState};
pub use calibration::{CalibrationData, CalibrationManager, GeometrySnapshot};
pub use array::FlexibleTransducerArray;

// Re-export at parent level for backward compatibility
pub use self::array::FlexibleTransducerArray as FlexibleTransducer;