//! Sensor-Specific Beamforming Interface
//!
//! Provides domain-layer beamforming operations that are tightly coupled to
//! specific sensor geometries and hardware characteristics. This serves as
//! the accessor layer between domain sensor concerns and analysis algorithms.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::sensor::grid_sampling::GridSensorSet;
use crate::domain::sensor::localization::array::SensorArray;
use crate::domain::signal::window as signal_window;
use ndarray::Array2;
use num_complex::Complex;

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
    /// Applies element-wise windowing to the delay matrix to reduce side lobes
    /// and improve image quality. The window is applied across the sensor dimension
    /// (rows of the delay matrix), with each column (image point) windowed independently.
    ///
    /// # Mathematical Basis
    ///
    /// For each image point $p$, the windowed delays are:
    /// $$ d'_i(p) = w_i \cdot d_i(p) $$
    /// where $w_i$ is the window coefficient for sensor $i$ and $d_i(p)$ is the
    /// geometric delay from sensor $i$ to point $p$.
    ///
    /// # Arguments
    ///
    /// * `delays` - Delay matrix (n_sensors × n_image_points)
    /// * `window_type` - Type of apodization window to apply
    ///
    /// # Returns
    ///
    /// Windowed delay matrix with same dimensions as input
    pub fn apply_windowing(
        &self,
        delays: &Array2<f64>,
        window_type: WindowType,
    ) -> KwaversResult<Array2<f64>> {
        let n_sensors = delays.nrows();

        // Convert WindowType to signal_window::WindowType
        let signal_window_type = match window_type {
            WindowType::Hanning => signal_window::WindowType::Hann,
            WindowType::Hamming => signal_window::WindowType::Hamming,
            WindowType::Blackman => signal_window::WindowType::Blackman,
            WindowType::Rectangular => signal_window::WindowType::Rectangular,
        };

        // Generate window coefficients for the sensor array (symmetric=true for apodization)
        let window_coeffs = signal_window::get_win(signal_window_type, n_sensors, true);

        // Apply window to each column (image point) independently
        let mut windowed_delays = delays.clone();
        for mut col in windowed_delays.columns_mut() {
            for (sensor_idx, window_coeff) in window_coeffs.iter().enumerate() {
                col[sensor_idx] *= window_coeff;
            }
        }

        Ok(windowed_delays)
    }

    /// Calculate steering vectors for adaptive beamforming
    ///
    /// This provides hardware-specific steering that accounts for
    /// element directivity patterns and array manifold characteristics.
    ///
    /// # Mathematical Basis
    ///
    /// Calculates array manifold vector $ \mathbf{v}(\theta, \phi) $:
    /// $$ \mathbf{v}(\theta, \phi) = [e^{-j \mathbf{k}^T \mathbf{p}_1}, \dots, e^{-j \mathbf{k}^T \mathbf{p}_N}]^T $$
    ///
    /// where $\mathbf{k}$ is the wavenumber vector for direction $(\theta, \phi)$.
    ///
    /// # Arguments
    ///
    /// * `angles` - List of (theta, phi) pairs in radians
    /// * `frequency` - Signal frequency in Hz
    /// * `sound_speed` - Speed of sound in m/s
    ///
    /// # Returns
    ///
    /// Complex steering matrix where each column corresponds to a direction in `angles`.
    /// Dimensions: (n_sensors, n_angles).
    pub fn calculate_steering(
        &self,
        angles: &[(f64, f64)],
        frequency: f64,
        sound_speed: f64,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        if frequency <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Frequency must be positive, got {}",
                frequency
            )));
        }
        if sound_speed <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Sound speed must be positive, got {}",
                sound_speed
            )));
        }

        let n_sensors = self.sensor_positions.len();
        let n_angles = angles.len();
        let mut steering_matrix = Array2::zeros((n_sensors, n_angles));

        for (idx, &(theta, phi)) in angles.iter().enumerate() {
            // Convert spherical (theta, phi) to Cartesian direction unit vector
            // Convention:
            // x = sin(theta) * cos(phi)
            // y = sin(theta) * sin(phi)
            // z = cos(theta)
            let sin_theta = theta.sin();
            let x = sin_theta * phi.cos();
            let y = sin_theta * phi.sin();
            let z = theta.cos();
            let direction = [x, y, z];

            // Use the shared SteeringVector logic from steering.rs
            // Note: compute_plane_wave returns Array1<Complex<f64>>
            let vector = super::steering::SteeringVector::compute_plane_wave(
                direction,
                frequency,
                &self.sensor_positions,
                sound_speed,
            )?;

            // Assign to column
            steering_matrix.column_mut(idx).assign(&vector);
        }

        Ok(steering_matrix)
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

/// Windowing functions for sensor array apodization
///
/// Apodization windows reduce side lobes in beamformed images by tapering
/// the contribution of elements toward the edges of the array aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hanning window (smooth, good side lobe suppression)
    /// $ w(n) = 0.5 - 0.5 \cos(2\pi n / (N-1)) $
    Hanning,
    /// Hamming window (similar to Hanning, slightly higher side lobes)
    /// $ w(n) = 0.54 - 0.46 \cos(2\pi n / (N-1)) $
    Hamming,
    /// Blackman window (excellent side lobe suppression)
    /// $ w(n) = 0.42 - 0.5 \cos(2\pi n / (N-1)) + 0.08 \cos(4\pi n / (N-1)) $
    Blackman,
    /// Rectangular window (no windowing, uniform weighting)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
    use crate::domain::sensor::localization::Position;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    fn create_test_array(n_sensors: usize) -> SensorArray {
        let sensors: Vec<Sensor> = (0..n_sensors)
            .map(|i| {
                let position = Position {
                    x: i as f64 * 0.001,
                    y: 0.0,
                    z: 0.0,
                };
                Sensor::new(i, position)
            })
            .collect();
        SensorArray::new(sensors, 1540.0, ArrayGeometry::Linear)
    }

    #[test]
    fn test_windowing_preserves_dimensions() {
        let array = create_test_array(8);
        let beamformer = SensorBeamformer::new(array, 1e6);

        let delays = Array2::ones((8, 100));

        for window_type in [
            WindowType::Hanning,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Rectangular,
        ] {
            let windowed = beamformer.apply_windowing(&delays, window_type).unwrap();
            assert_eq!(windowed.shape(), delays.shape());
        }
    }

    #[test]
    fn test_rectangular_window_is_identity() {
        let array = create_test_array(8);
        let beamformer = SensorBeamformer::new(array, 1e6);

        let delays = Array2::from_shape_fn((8, 100), |(i, j)| i as f64 + j as f64 * 0.1);

        let windowed = beamformer
            .apply_windowing(&delays, WindowType::Rectangular)
            .unwrap();

        // Rectangular window should be all ones, so output equals input
        for i in 0..8 {
            for j in 0..100 {
                assert_relative_eq!(windowed[[i, j]], delays[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_window_reduces_edge_elements() {
        let array = create_test_array(16);
        let beamformer = SensorBeamformer::new(array, 1e6);

        let delays = Array2::ones((16, 50));

        for window_type in [
            WindowType::Hanning,
            WindowType::Hamming,
            WindowType::Blackman,
        ] {
            let windowed = beamformer.apply_windowing(&delays, window_type).unwrap();

            // Check that edge elements are reduced (window should taper to ~0 at edges)
            // For symmetric windows, first and last elements should be equal
            assert_relative_eq!(windowed[[0, 0]], windowed[[15, 0]], epsilon = 1e-10);

            // Edge elements should be smaller than center elements
            let center_val = windowed[[8, 0]];
            let edge_val = windowed[[0, 0]];
            assert!(
                edge_val < center_val,
                "Edge value {} should be less than center value {} for {:?}",
                edge_val,
                center_val,
                window_type
            );
        }
    }

    #[test]
    fn test_hanning_window_has_zero_endpoints() {
        let array = create_test_array(8);
        let beamformer = SensorBeamformer::new(array, 1e6);

        let delays = Array2::ones((8, 10));
        let windowed = beamformer
            .apply_windowing(&delays, WindowType::Hanning)
            .unwrap();

        // Hanning window should have values very close to zero at endpoints
        assert!(windowed[[0, 0]].abs() < 1e-10);
        assert!(windowed[[7, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_windowing_applied_per_column() {
        let array = create_test_array(4);
        let beamformer = SensorBeamformer::new(array, 1e6);

        // Create delays with different values per column
        let mut delays = Array2::zeros((4, 3));
        delays.column_mut(0).fill(1.0);
        delays.column_mut(1).fill(2.0);
        delays.column_mut(2).fill(3.0);

        let windowed = beamformer
            .apply_windowing(&delays, WindowType::Hamming)
            .unwrap();

        // Each column should be scaled by the same window coefficients
        // So ratios within columns should be preserved
        for col_idx in 0..3 {
            for row_idx in 0..4 {
                let original_val = delays[[row_idx, col_idx]];
                let windowed_val = windowed[[row_idx, col_idx]];
                let ratio = windowed_val / original_val;

                // This ratio should be the same for all columns (same row)
                if col_idx > 0 {
                    let prev_ratio =
                        windowed[[row_idx, col_idx - 1]] / delays[[row_idx, col_idx - 1]];
                    assert_relative_eq!(ratio, prev_ratio, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_blackman_window_has_better_sidelobe_suppression() {
        let array = create_test_array(32);
        let beamformer = SensorBeamformer::new(array, 1e6);

        let delays = Array2::ones((32, 1));

        let hanning = beamformer
            .apply_windowing(&delays, WindowType::Hanning)
            .unwrap();
        let blackman = beamformer
            .apply_windowing(&delays, WindowType::Blackman)
            .unwrap();

        // Blackman should taper more aggressively at edges
        // (lower values at edge elements compared to Hanning)
        let edge_idx = 1; // Near edge but not exactly at zero
        assert!(
            blackman[[edge_idx, 0]] < hanning[[edge_idx, 0]],
            "Blackman window should suppress edge more than Hanning"
        );
    }

    #[test]
    fn test_windowing_with_zero_delays() {
        let array = create_test_array(8);
        let beamformer = SensorBeamformer::new(array, 1e6);

        let delays = Array2::zeros((8, 10));
        let windowed = beamformer
            .apply_windowing(&delays, WindowType::Hanning)
            .unwrap();

        // Windowing zeros should produce zeros
        for i in 0..8 {
            for j in 0..10 {
                assert_eq!(windowed[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn test_processing_params_f_number() {
        let params = SensorProcessingParams {
            n_sensors: 64,
            sampling_frequency: 1e6,
            element_spacing: 0.3e-3,
            array_aperture: 19.2e-3,
        };

        let focal_length = 50e-3;
        let f_num = params.f_number(focal_length);
        assert_relative_eq!(f_num, 50e-3 / 19.2e-3, epsilon = 1e-10);
    }

    #[test]
    fn test_processing_params_max_spatial_frequency() {
        let params = SensorProcessingParams {
            n_sensors: 64,
            sampling_frequency: 1e6,
            element_spacing: 0.3e-3,
            array_aperture: 19.2e-3,
        };

        let sound_speed = 1540.0;
        let max_freq = params.max_spatial_frequency(sound_speed);
        let expected = sound_speed / (2.0 * 0.3e-3);
        assert_relative_eq!(max_freq, expected, epsilon = 1e-6);
    }
}
