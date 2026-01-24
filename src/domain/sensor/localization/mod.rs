// sensor/localization/mod.rs - Unified acoustic source localization

pub mod algorithms;
pub mod array;
pub mod beamforming_search;
pub mod multilateration;
pub mod tdoa;
pub mod triangulation;

use crate::analysis::signal_processing::beamforming::domain_processor::BeamformingProcessor;
use crate::core::error::KwaversResult;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

// Re-export main types
pub use algorithms::{LocalizationMethod, LocalizationProcessor as AlgorithmProcessor};
pub use array::{ArrayGeometry, SensorArray};

// Beamforming-based localization: keep orchestration in localization, keep numerics in beamforming SSOT.
//
// NOTE: method naming must follow field jargon:
// - A (transient): SRP-DAS (Steered Response Power using time-domain DAS)
// - B (narrowband/adaptive): Steered MVDR/Capon spatial spectrum
pub use beamforming_search::{
    BeamformSearch, LocalizationBeamformSearchConfig, LocalizationBeamformingMethod, SearchGrid,
};
pub use multilateration::trilateration::{TrilaterationResult, TrilaterationSolver};
pub use multilateration::{MultilaterationMethod, MultilaterationSolver};
pub use tdoa::{TDOAMeasurement, TDOAProcessor};
pub use triangulation::Triangulator;

/// Position in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position {
    /// Create new position
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Distance to another position
    #[must_use]
    pub fn distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to array
    #[must_use]
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// From array
    #[must_use]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Frequency of signal \[Hz\] (for beamforming)
    pub frequency: f64,
    /// Search radius for grid-based methods \[m\]
    pub search_radius: Option<f64>,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1500.0, // Water
            max_iterations: 100,
            tolerance: 1e-6,
            use_gpu: false,
            method: LocalizationMethod::TDOA,
            frequency: 1e6,      // 1 MHz default
            search_radius: None, // Auto-determined
        }
    }
}

/// Dedicated input type for beamforming-based localization.
///
/// This is required because SSOT-compliant beamforming requires raw time-series data:
/// shape `(n_sensors, 1, n_samples)`. Scalar per-sensor measurements are insufficient
/// for time-domain DAS/MVDR or any covariance/subspace method.
#[derive(Debug, Clone)]
pub struct BeamformingLocalizationInput {
    /// Raw sensor time-series data shaped `(n_sensors, 1, n_samples)`.
    pub sensor_data: Array3<f64>,
    /// Sampling frequency in Hz (must match the acquisition used to generate `sensor_data`).
    pub sampling_frequency: f64,
}

impl BeamformingLocalizationInput {
    /// Validate invariants for the beamforming-localization input.
    pub fn validate(&self, expected_sensors: usize) -> KwaversResult<()> {
        let (n_sensors, channels, n_samples) = self.sensor_data.dim();
        if n_sensors != expected_sensors {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "BeamformingLocalizationInput: sensor_data n_sensors ({n_sensors}) does not match expected ({expected_sensors})"
            )));
        }
        if channels != 1 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "BeamformingLocalizationInput: expected channels=1, got {channels}"
            )));
        }
        if n_samples == 0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "BeamformingLocalizationInput: n_samples must be > 0".to_string(),
            ));
        }
        if !self.sampling_frequency.is_finite() || self.sampling_frequency <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "BeamformingLocalizationInput: sampling_frequency must be finite and > 0"
                    .to_string(),
            ));
        }
        Ok(())
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
    #[must_use]
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

    /// Beamforming-based localization using raw time-series data (SSOT compliant).
    ///
    /// This is the dedicated API that allows deleting the redundant beamforming implementation
    /// previously living in `sensor::localization`.
    ///
    /// Internals:
    /// - Constructs `BeamformingCoreConfig` from `LocalizationConfig`.
    /// - Builds a shared `BeamformingProcessor` using sensor positions.
    /// - Runs `BeamformSearch` using a caller-supplied `LocalizationBeamformSearchConfig`.
    pub fn localize_beamforming(
        &self,
        input: &BeamformingLocalizationInput,
        search_cfg: LocalizationBeamformSearchConfig,
    ) -> KwaversResult<LocalizationResult> {
        use std::time::Instant;

        let start = Instant::now();

        let expected = self.sensor_array.num_sensors();
        input.validate(expected)?;
        search_cfg.validate()?;

        // Ensure sampling frequency consistency between policy core and input.
        if (search_cfg.core.sampling_frequency - input.sampling_frequency).abs() > 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "localize_beamforming: search_cfg.core.sampling_frequency must equal input.sampling_frequency to keep frequency-to-sample mapping consistent"
                    .to_string(),
            ));
        }

        let sensor_positions: Vec<[f64; 3]> = self
            .sensor_array
            .get_sensor_positions()
            .iter()
            .map(|p| p.to_array())
            .collect();

        let beamformer = BeamformingProcessor::new(
            search_cfg.core.clone(),
            sensor_positions,
        );

        let search = crate::domain::sensor::localization::beamforming_search::BeamformSearch::new(
            beamformer, search_cfg,
        )?;

        let centroid = self.sensor_array.centroid().to_array();
        let position = search.search(centroid, &input.sensor_data)?;

        let computation_time = start.elapsed().as_secs_f64();

        Ok(LocalizationResult {
            position,
            // Placeholder uncertainty model removed: consumers should compute uncertainty
            // using residuals or a second-pass refinement strategy. For now we report zeros.
            uncertainty: Position::new(0.0, 0.0, 0.0),
            confidence: 0.0,
            method: LocalizationMethod::Beamforming,
            computation_time,
        })
    }

    /// Update configuration
    pub fn set_config(&mut self, config: LocalizationConfig) {
        self.config = config;
    }

    /// Get sensor array
    #[must_use]
    pub fn sensor_array(&self) -> &SensorArray {
        &self.sensor_array
    }
}
