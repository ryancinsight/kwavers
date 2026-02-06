//! Configuration types for flexible transducer arrays
//!
//! This module defines the configuration structures and enums for flexible
//! transducer arrays, following SSOT and SOLID principles.

use serde::{Deserialize, Serialize};

/// Configuration for flexible transducer arrays
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlexibleTransducerConfig {
    /// Number of elements in the array
    pub num_elements: usize,
    /// Nominal element spacing when flat (m)
    pub nominal_spacing: f64,
    /// Element dimensions [width, height] (m)
    pub element_size: [f64; 2],
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Flexibility parameters
    pub flexibility: FlexibilityModel,
    /// Calibration method for geometry estimation
    pub calibration_method: CalibrationMethod,
    /// Update frequency for geometry tracking (Hz)
    pub tracking_frequency: f64,
}

impl Default for FlexibleTransducerConfig {
    fn default() -> Self {
        Self {
            num_elements: 128,
            nominal_spacing: 0.3e-3, // λ/2 at 2.5 MHz
            element_size: [0.25e-3, 10e-3],
            frequency: 2.5e6,
            flexibility: FlexibilityModel::Elastic {
                young_modulus: 2e9, // 2 GPa for flexible materials
                poisson_ratio: 0.3,
                thickness: 0.5e-3, // 0.5 mm
            },
            calibration_method: CalibrationMethod::SelfCalibration {
                reference_reflectors: vec![[0.0, 0.0, 50e-3]],
                calibration_interval: 1.0, // 1 second
            },
            tracking_frequency: 100.0, // 100 Hz
        }
    }
}

/// Flexibility models for different transducer types
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum FlexibilityModel {
    /// Rigid array (no deformation)
    Rigid,
    /// Elastic deformation model
    Elastic {
        young_modulus: f64, // Pa
        poisson_ratio: f64,
        thickness: f64, // m
    },
    /// Fluid-filled flexible array
    FluidFilled {
        fluid_bulk_modulus: f64, // Pa
        membrane_tension: f64,   // N/m
    },
}

/// Calibration methods for geometry estimation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum CalibrationMethod {
    /// Self-calibration using known reflectors
    SelfCalibration {
        reference_reflectors: Vec<[f64; 3]>,
        calibration_interval: f64, // seconds
    },
    /// External tracking system
    ExternalTracking {
        tracking_system: TrackingSystem,
        measurement_noise: f64, // m
    },
    /// Image-based calibration
    ImageBased {
        feature_detection_threshold: f64,
        correlation_window_size: usize,
    },
    /// Hybrid approach combining multiple methods
    Hybrid {
        primary_method: Box<CalibrationMethod>,
        fallback_method: Box<CalibrationMethod>,
    },
}

/// External tracking system types
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum TrackingSystem {
    /// Optical tracking (e.g., `OptiTrack`, Vicon)
    Optical {
        marker_positions: Vec<[f64; 3]>,
        camera_count: usize,
    },
    /// Electromagnetic tracking (e.g., Polhemus, NDI Aurora)
    Electromagnetic {
        sensor_positions: Vec<[f64; 3]>,
        field_strength: f64, // Tesla
    },
    /// Inertial measurement units
    IMU {
        sensor_count: usize,
        sampling_rate: f64, // Hz
    },
}
