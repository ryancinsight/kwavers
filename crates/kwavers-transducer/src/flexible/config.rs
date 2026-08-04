//! Configuration types for flexible transducer arrays
//!
//! This module defines the configuration structures and enums for flexible
//! transducer arrays, following SSOT and SOLID principles.

use aequitas::systems::si::quantities::{Frequency, Length, Pressure, SpringStiffness, Time};
use aequitas::systems::si::units::{Hertz, Meter, Pascal, Second};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use serde::{Deserialize, Serialize};

/// Configuration for flexible transducer arrays.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlexibleTransducerConfig {
    /// Number of elements in the array.
    pub num_elements: usize,
    /// Nominal element centre-to-centre spacing when the array is flat.
    pub nominal_spacing: Length<f64>,
    /// Element dimensions `[width, height]`.
    pub element_size: [Length<f64>; 2],
    /// Operating centre frequency.
    pub frequency: Frequency<f64>,
    /// Flexibility parameters.
    pub flexibility: FlexibilityModel,
    /// Calibration method for geometry estimation.
    pub calibration_method: CalibrationMethod,
    /// Update rate for geometry tracking.
    pub tracking_frequency: Frequency<f64>,
}

impl Default for FlexibleTransducerConfig {
    fn default() -> Self {
        Self {
            num_elements: 128,
            nominal_spacing: Length::from_unit::<Meter>(0.3e-3), // λ/2 at 2.5 MHz
            element_size: [
                Length::from_unit::<Meter>(0.25e-3),
                Length::from_unit::<Meter>(10e-3),
            ],
            frequency: Frequency::from_unit::<Hertz>(2.5 * MHZ_TO_HZ),
            flexibility: FlexibilityModel::Elastic {
                young_modulus: Pressure::from_unit::<Pascal>(2e9), // 2 GPa for flexible materials
                poisson_ratio: 0.3,
                thickness: Length::from_unit::<Meter>(0.5e-3), // 0.5 mm
            },
            calibration_method: CalibrationMethod::SelfCalibration {
                reference_reflectors: vec![[0.0, 0.0, 50e-3]],
                calibration_interval: Time::from_unit::<Second>(1.0), // 1 second
            },
            tracking_frequency: Frequency::from_unit::<Hertz>(100.0), // 100 Hz
        }
    }
}

/// Flexibility models for different transducer types.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum FlexibilityModel {
    /// Rigid array (no deformation).
    Rigid,
    /// Elastic deformation model.
    Elastic {
        /// Young's modulus of the substrate.
        young_modulus: Pressure<f64>,
        /// Poisson ratio (dimensionless).
        poisson_ratio: f64,
        /// Substrate thickness.
        thickness: Length<f64>,
    },
    /// Fluid-filled flexible array.
    FluidFilled {
        /// Bulk modulus of the fill fluid.
        fluid_bulk_modulus: Pressure<f64>,
        /// Membrane tension.
        membrane_tension: SpringStiffness<f64>,
    },
}

/// Calibration methods for geometry estimation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum CalibrationMethod {
    /// Self-calibration using known reflectors.
    SelfCalibration {
        /// Mesh coordinates `[x, y, z]` of reference reflectors in metres.
        ///
        /// Kept as raw `[f64; 3]` — passed directly to the mesh layer.
        reference_reflectors: Vec<[f64; 3]>,
        /// Time between calibration updates.
        calibration_interval: Time<f64>,
    },
    /// External tracking system.
    ExternalTracking {
        /// Tracking system configuration.
        tracking_system: TrackingSystem,
        /// Estimated position measurement noise (1-σ).
        measurement_noise: Length<f64>,
    },
    /// Image-based calibration.
    ImageBased {
        /// Detection threshold (dimensionless).
        feature_detection_threshold: f64,
        /// Correlation window half-width in samples.
        correlation_window_size: usize,
    },
    /// Hybrid approach combining multiple methods.
    Hybrid {
        /// Primary calibration method.
        primary_method: Box<Self>,
        /// Fallback calibration method.
        fallback_method: Box<Self>,
    },
}

/// External tracking system types.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum TrackingSystem {
    /// Optical tracking (e.g., OptiTrack, Vicon).
    Optical {
        /// Marker mesh coordinates `[x, y, z]` in metres.
        marker_positions: Vec<[f64; 3]>,
        /// Number of cameras.
        camera_count: usize,
    },
    /// Electromagnetic tracking (e.g., Polhemus, NDI Aurora).
    Electromagnetic {
        /// Sensor mesh coordinates `[x, y, z]` in metres.
        sensor_positions: Vec<[f64; 3]>,
        /// Magnetic field strength in Tesla.
        ///
        /// Kept as raw `f64` — Aequitas does not yet define a
        /// `MagneticFluxDensity` quantity.
        field_strength: f64,
    },
    /// Inertial measurement units.
    IMU {
        /// Number of sensors.
        sensor_count: usize,
        /// IMU sampling rate.
        sampling_rate: Frequency<f64>,
    },
}
