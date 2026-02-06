//! Flexible transducer arrays with real-time geometry tracking
//!
//! Modular implementation following SOLID principles

pub mod array;
pub mod calibration;
pub mod config;
pub mod geometry;

pub use array::FlexibleTransducerArray;
pub use calibration::{CalibrationData, CalibrationManager, GeometrySnapshot};
pub use config::{CalibrationMethod, FlexibilityModel, FlexibleTransducerConfig, TrackingSystem};
pub use geometry::{DeformationState, GeometryState};

// Re-export at parent level for backward compatibility
pub use self::array::FlexibleTransducerArray as FlexibleTransducer;
