//! Sensor-Specific Beamforming Interface
//!
//! Provides domain-layer beamforming operations that are tightly coupled to
//! specific sensor geometries and hardware characteristics. This serves as
//! the accessor layer between domain sensor concerns and analysis algorithms.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::sensor::grid_sampling::GridSensorSet;
use crate::domain::sensor::localization::array::SensorArray;
use ndarray::Array2;

/// Sensor-specific beamforming operations tied to hardware geometry
#[derive(Debug, Clone)]
pub struct SensorBeamformer {
    /// Sensor array geometry
    sensor_array: SensorArray,
    /// Grid sensor set for sampling (if applicable)
    grid_sensors: Option<GridSensorSet>,
    /// Cached sensor positions for performance
    sensor_positions: Vec<[f64; 3]>,
    /// Sampling frequency for time-domain calculations
    sampling_frequency: f64,
}

impl SensorBeamformer {
    /// Create a new sensor beamformer for the given array
    #[must_use]
    pub fn new(sensor_array: SensorArray, sampling_frequency: f64) -> Self {
        let sensor_positions = sensor_array
            .get_sensor_positions()
            .into_iter()
            .map(|pos| [pos.x, pos.y, pos.z])
            .collect();

        Self {
            sensor_array,
            grid_sensors: None,
            sensor_positions,
            sampling_frequency,
        }
    }

    /// Create a sensor beamformer with grid sampling capability
    #[must_use]
    pub fn with_grid_sensors(
        sensor_array: SensorArray,
        grid_sensors: GridSensorSet,
        sampling_frequency: f64,
    ) -> Self {
        let sensor_positions = sensor_array
            .get_sensor_positions()
            .into_iter()
            .map(|pos| [pos.x, pos.y, pos.z])
            .collect();

        Self {
            sensor_array,
            grid_sensors: Some(grid_sensors),
            sensor_positions,
            sampling_frequency,
        }
    }

    /// Get sensor positions as array
    #[must_use]
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.sensor_positions
    }

    /// Calculate geometric delays from sensors to image points
    ///
    /// This is hardware-specific as it depends on the exact sensor positions
    /// and assumes a specific coordinate system and timing reference.
    ///
    /// # TODO: INCOMPLETE IMPLEMENTATION
    /// This method returns zero-filled placeholder values. Complete implementation requires:
    /// - Geometric distance calculation from each sensor to each image grid point
    /// - Time-of-flight calculation using sound_speed parameter
    /// - Reference time correction based on array geometry
    /// - Validation against physical constraints (causality, array aperture limits)
    /// See backlog.md item #6 for specifications (6-8 hour effort estimate)
    pub fn calculate_delays(
        &self,
        image_grid: &Grid,
        sound_speed: f64,
    ) -> KwaversResult<Array2<f64>> {
        if sound_speed <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Sound speed must be positive, got {}",
                sound_speed
            )));
        }

        let (nx, ny, nz) = image_grid.dimensions();
        let (dx, dy, dz) = image_grid.spacing();
        let origin = image_grid.origin;

        let mut delays = Array2::zeros((self.sensor_positions.len(), image_grid.size()));

        for (sensor_idx, sensor_pos) in self.sensor_positions.iter().enumerate() {
            let mut grid_idx = 0;
            // Iterate in memory order (row-major for C/ndarray default)
            // ndarray::Array3 [i, j, k] corresponds to idx = i*(ny*nz) + j*nz + k
            for i in 0..nx {
                let x = origin[0] + i as f64 * dx;
                for j in 0..ny {
                    let y = origin[1] + j as f64 * dy;
                    for k in 0..nz {
                        let z = origin[2] + k as f64 * dz;

                        let dist = ((x - sensor_pos[0]).powi(2)
                            + (y - sensor_pos[1]).powi(2)
                            + (z - sensor_pos[2]).powi(2))
                        .sqrt();

                        delays[[sensor_idx, grid_idx]] = dist / sound_speed;
                        grid_idx += 1;
                    }
                }
            }
        }

        Ok(delays)
    }

    /// Apply apodization window specific to this sensor array
    ///
    /// Windowing may depend on array geometry, element directivity, and
    /// hardware-specific constraints.
    ///
    /// # TODO: INCOMPLETE IMPLEMENTATION
    /// This method returns unmodified input - no actual windowing applied. Complete implementation requires:
    /// - Hanning window implementation (raised cosine)
    /// - Hamming window implementation (modified raised cosine)
    /// - Blackman window implementation (optimal side lobe suppression)
    /// - Rectangular window (pass-through, current behavior)
    /// - Window coefficient calculation based on array geometry
    /// - Element-wise multiplication of delays with window coefficients
    /// See backlog.md item #6 for specifications (6-8 hour effort estimate)
    pub fn apply_windowing(
        &self,
        delays: &Array2<f64>,
        _window_type: WindowType,
    ) -> KwaversResult<Array2<f64>> {
        // TODO: Replace with proper windowing implementation
        // Current: Returns unmodified input - no apodization applied (INVALID for production)
        Ok(delays.clone())
    }

    /// Calculate steering vectors for adaptive beamforming
    ///
    /// This provides hardware-specific steering that accounts for
    /// element directivity patterns and array manifold characteristics.
    ///
    /// # TODO: INCOMPLETE IMPLEMENTATION
    /// This method returns identity matrix placeholder. Complete implementation requires:
    /// - Array manifold calculation: exp(-j * 2π * f * delay(θ,φ) / c)
    /// - Phase delays for each sensor element at given angles
    /// - Frequency-dependent steering vector computation
    /// - Element directivity pattern incorporation
    /// - Validation against physical array manifold properties (unitary, Hermitian symmetry)
    /// Mathematical basis: Van Trees (2002) "Optimum Array Processing", Chapter 2
    /// See backlog.md item #6 for specifications (6-8 hour effort estimate)
    pub fn calculate_steering(
        &self,
        _angles: &[f64],
        _frequency: f64,
    ) -> KwaversResult<Array2<f64>> {
        // TODO: Replace with proper steering vector calculation
        // Current: Returns identity matrix - INVALID for adaptive beamforming (no actual steering)
        Ok(Array2::eye(self.sensor_positions.len()))
    }

    /// Get sensor-specific processing parameters
    #[must_use]
    pub fn processing_params(&self) -> SensorProcessingParams {
        SensorProcessingParams {
            n_sensors: self.sensor_positions.len(),
            sampling_frequency: self.sampling_frequency,
            element_spacing: self.calculate_element_spacing(),
            array_aperture: self.calculate_aperture(),
        }
    }

    /// Calculate average element spacing (domain-specific metric)
    #[must_use]
    fn calculate_element_spacing(&self) -> f64 {
        if self.sensor_positions.len() < 2 {
            return 0.0;
        }

        // Calculate average spacing between adjacent elements
        let mut total_spacing = 0.0;
        let mut count = 0;

        for i in 0..self.sensor_positions.len() - 1 {
            let pos1 = self.sensor_positions[i];
            let pos2 = self.sensor_positions[i + 1];
            let spacing = ((pos1[0] - pos2[0]).powi(2)
                + (pos1[1] - pos2[1]).powi(2)
                + (pos1[2] - pos2[2]).powi(2))
            .sqrt();
            total_spacing += spacing;
            count += 1;
        }

        if count > 0 {
            total_spacing / count as f64
        } else {
            0.0
        }
    }

    /// Calculate array aperture (domain-specific metric)
    #[must_use]
    fn calculate_aperture(&self) -> f64 {
        if self.sensor_positions.is_empty() {
            return 0.0;
        }

        // Find min/max coordinates
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;

        for pos in &self.sensor_positions {
            min_x = min_x.min(pos[0]);
            max_x = max_x.max(pos[0]);
        }

        max_x - min_x
    }
}

/// Windowing functions for sensor arrays
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hanning window (smooth, good side lobe suppression)
    Hanning,
    /// Hamming window (similar to Hanning, slightly higher side lobes)
    Hamming,
    /// Blackman window (excellent side lobe suppression)
    Blackman,
    /// Rectangular window (no windowing)
    Rectangular,
}

/// Sensor-specific processing parameters
#[derive(Debug, Clone)]
pub struct SensorProcessingParams {
    /// Number of sensors in the array
    pub n_sensors: usize,
    /// Sampling frequency in Hz
    pub sampling_frequency: f64,
    /// Average element spacing in meters
    pub element_spacing: f64,
    /// Array aperture in meters
    pub array_aperture: f64,
}

impl SensorProcessingParams {
    /// Calculate the array's F-number (focal_length / aperture)
    #[must_use]
    pub fn f_number(&self, focal_length: f64) -> f64 {
        focal_length / self.array_aperture
    }

    /// Calculate maximum frequency for spatial Nyquist sampling
    #[must_use]
    pub fn max_spatial_frequency(&self, sound_speed: f64) -> f64 {
        sound_speed / (2.0 * self.element_spacing)
    }
}
