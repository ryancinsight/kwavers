// sensor/localization/mod.rs - Unified acoustic source localization

pub mod algorithms;
pub mod array;
pub mod beamforming;
pub mod tdoa;
pub mod triangulation;

use crate::error::KwaversResult;
use serde::{Deserialize, Serialize};

// Re-export main types
pub use algorithms::{LocalizationMethod, LocalizationProcessor as AlgorithmProcessor};
pub use array::{ArrayGeometry, SensorArray};
pub use beamforming::Beamformer;
pub use tdoa::{TDOAMeasurement, TDOAProcessor};
pub use triangulation::Triangulator;

/// Position in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position {
    /// Create new position
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Distance to another position
    pub fn distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// From array
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

/// Localization result with uncertainty
#[derive(Debug, Clone)]
pub struct LocalizationResult {
    /// Estimated position
    pub position: Position,
    /// Position uncertainty (standard deviation)
    pub uncertainty: Position,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Method used
    pub method: LocalizationMethod,
    /// Computation time in seconds
    pub computation_time: f64,
}

/// Localization configuration
#[derive(Debug, Clone, Serialize, Deserialize]
pub struct LocalizationConfig {
    /// Speed of sound in medium [m/s]
    pub sound_speed: f64,
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Localization method
    pub method: LocalizationMethod,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1500.0, // Water
            max_iterations: 100,
            tolerance: 1e-6,
            use_gpu: false,
            method: LocalizationMethod::TDOA,
        }
    }
}

/// Main localization processor
#[derive(Debug)]
pub struct LocalizationProcessor {
    config: LocalizationConfig,
    sensor_array: SensorArray,
}

impl LocalizationProcessor {
    /// Create new processor
    pub fn new(config: LocalizationConfig, sensor_array: SensorArray) -> Self {
        Self {
            config,
            sensor_array,
        }
    }

    /// Localize source from measurements
    pub fn localize(&self, measurements: &[f64]) -> KwaversResult<LocalizationResult> {
        let processor = AlgorithmProcessor::from_method(self.config.method);
        processor.localize(&self.sensor_array, measurements, &self.config)
    }

    /// Update configuration
    pub fn set_config(&mut self, config: LocalizationConfig) {
        self.config = config;
    }

    /// Get sensor array
    pub fn sensor_array(&self) -> &SensorArray {
        &self.sensor_array
    }
}
