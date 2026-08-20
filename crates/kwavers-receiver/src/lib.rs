#![doc = include_str!("../README.md")]

pub mod array;
pub mod grid_sampling;
pub mod point;
pub mod recorder;
pub mod sonoluminescence;

// Canonical high-level probe set (supports both acoustics + optics).
pub use grid_sampling::{GridPoint, GridSensorSet};

// Sensor array types (hardware geometry — SSOT for sensor positions)
pub use array::{Position, Sensor, SensorArray, SensorArrayGeometry};

// Point sensors for hydrophone-equivalent arbitrary position sampling
pub use point::{PointSensor, PointSensorConfig};
