//! Flexible transducer arrays with real-time geometry tracking
//!
//! Modular implementation following SOLID principles

mod config;
mod geometry;
mod calibration;
mod array;

pub use config::{FlexibleTransducerConfig, FlexibilityModel};
pub use geometry::{GeometryState, DeformationState};
pub use calibration::{CalibrationMethod, TrackingSystem};
pub use array::FlexibleTransducerArray;

// Re-export at parent level for backward compatibility
pub use self::array::FlexibleTransducerArray as FlexibleTransducer;