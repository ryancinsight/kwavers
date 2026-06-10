//! Flexible transducer arrays with real-time geometry tracking
//!
//! Modular implementation following SOLID principles

pub mod array;
pub mod beamforming;
pub mod calibration;
pub mod config;
pub mod geometry;

pub use array::FlexibleTransducerArray;
pub use beamforming::{
    cmut_flex_apodization, focusing_delays, per_element_curvature, steering_delays,
};
pub use calibration::{CalibrationData, CalibrationManager, GeometrySnapshot};
pub use config::{CalibrationMethod, FlexibilityModel, FlexibleTransducerConfig, TrackingSystem};
pub use geometry::{DeformationState, GeometryState};
