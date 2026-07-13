//! `SensorBeamformer` — geometry-aware delay, apodization, and steering for sensor arrays.

use super::types::{BeamformerWindowType, SensorProcessingParams};
use eunomia::Complex;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_receiver::array::SensorArray;
use kwavers_receiver::grid_sampling::GridSensorSet;
use kwavers_signal::window as signal_window;
use leto::Array2;

/// Sensor-specific beamforming interface tied to hardware array geometry.
#[derive(Debug, Clone)]
pub struct SensorBeamformer {
    /// Cached 3-D sensor positions `[x, y, z]` (m) derived from the sensor array at construction.
    sensor_positions: Vec<[f64; 3]>,
    /// Sampling frequency (Hz).
    sampling_frequency: f64,
}

impl SensorBeamformer {
    /// Construct from a sensor array and sampling frequency.
    #[must_use]
    pub fn new(sensor_array: SensorArray, sampling_frequency: f64) -> Self {
        let sensor_positions = sensor_array
            .get_sensor_positions()
            .into_iter()
            .map(|pos| [pos.x, pos.y, pos.z])
            .collect();

        Self {
            sensor_positions,
            sampling_frequency,
        }
    }

    /// Construct with an additional grid sensor set for volumetric sampling.
    ///
    /// `grid_sensors` is accepted for API symmetry; volumetric-sampling dispatch
    /// is a tracked backlog item.
    #[must_use]
    pub fn with_grid_sensors(
        sensor_array: SensorArray,
        _grid_sensors: GridSensorSet,
        sampling_frequency: f64,
    ) -> Self {
        let sensor_positions = sensor_array
            .get_sensor_positions()
            .into_iter()
            .map(|pos| [pos.x, pos.y, pos.z])
            .collect();

        Self {
            sensor_positions,
            sampling_frequency,
        }
    }

    /// Sensor positions as a slice of `[x, y, z]` triples (m).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.sensor_positions
    }

    /// Compute one-way time-of-flight delays from each sensor to each grid point.
    ///
    /// Returns an `(n_sensors × n_grid_points)` matrix where element `[s, p]` is
    /// the Euclidean distance from sensor `s` to grid point `p` divided by
    /// `sound_speed`.
    ///
    /// Grid points are enumerated in row-major (C) order:
    /// `idx = i*(ny*nz) + j*nz + k`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_delays(
        &self,
        image_grid: &Grid,
        sound_speed: f64,
    ) -> KwaversResult<Array2<f64>> {
        if sound_speed <= 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
                "Sound speed must be positive, got {}",
                sound_speed
            )));
        }

        let origin = image_grid.origin;
        let mut delays = Array2::zeros([self.sensor_positions.len(), image_grid.size()]);
        let x_coords: Vec<f64> = image_grid
            .coordinates(kwavers_grid::GridDimension::X)
            .collect();
        let y_coords: Vec<f64> = image_grid
            .coordinates(kwavers_grid::GridDimension::Y)
            .collect();
        let z_coords: Vec<f64> = image_grid
            .coordinates(kwavers_grid::GridDimension::Z)
            .collect();

        for (sensor_idx, sensor_pos) in self.sensor_positions.iter().enumerate() {
            let mut grid_idx = 0;
            for &x_rel in &x_coords {
                let x = origin[0] + x_rel;
                for &y_rel in &y_coords {
                    let y = origin[1] + y_rel;
                    for &z_rel in &z_coords {
                        let z = origin[2] + z_rel;
                        let dist = (z - sensor_pos[2])
                            .mul_add(
                                z - sensor_pos[2],
                                (y - sensor_pos[1])
                                    .mul_add(y - sensor_pos[1], (x - sensor_pos[0]).powi(2)),
                            )
                            .sqrt();
                        delays[[sensor_idx, grid_idx]] = dist / sound_speed;
                        grid_idx += 1;
                    }
                }
            }
        }

        Ok(delays)
    }

    /// Apply element-wise apodization window to a delay matrix.
    ///
    /// For each image point column `p`, each sensor row `s` is scaled by the
    /// symmetric window coefficient `w(s)`:
    /// ```text
    /// d'[s, p] = w(s) · d[s, p]
    /// ```
    ///
    /// # Arguments
    /// - `delays` — `(n_sensors × n_image_points)` delay matrix.
    /// - `window_type` — Apodization window to apply.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_windowing(
        &self,
        delays: &Array2<f64>,
        window_type: BeamformerWindowType,
    ) -> KwaversResult<Array2<f64>> {
        let n_sensors = delays.shape()[0];

        let signal_window_type = match window_type {
            BeamformerWindowType::Hanning => signal_window::SignalWindowType::Hann,
            BeamformerWindowType::Hamming => signal_window::SignalWindowType::Hamming,
            BeamformerWindowType::Blackman => signal_window::SignalWindowType::Blackman,
            BeamformerWindowType::Rectangular => signal_window::SignalWindowType::Rectangular,
        };

        let window_coeffs = signal_window::get_win(signal_window_type, n_sensors, true);

        let mut windowed_delays = delays.clone();
        let [_n_sensors, n_pts] = windowed_delays.shape();
        for pt in 0..n_pts {
            for (sensor_idx, window_coeff) in window_coeffs.iter().enumerate() {
                windowed_delays[[sensor_idx, pt]] *= window_coeff;
            }
        }

        Ok(windowed_delays)
    }

    /// Compute the complex array manifold (steering) matrix.
    ///
    /// Returns an `(n_sensors × n_angles)` matrix. Column `k` is the array manifold
    /// vector for direction `angles[k] = (θ, φ)`:
    /// ```text
    /// v(θ, φ) = [exp(−j kᵀ p₁), …, exp(−j kᵀ p_N)]ᵀ
    /// ```
    /// where **k** is the wavenumber vector derived from spherical convention:
    /// `(sin θ cos φ, sin θ sin φ, cos θ)`.
    ///
    /// # References
    /// - Van Trees, H. L. (2002): *Optimum Array Processing*, §2.4. Wiley.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calculate_steering(
        &self,
        angles: &[(f64, f64)],
        frequency: f64,
        sound_speed: f64,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        if frequency <= 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
                "Frequency must be positive, got {}",
                frequency
            )));
        }
        if sound_speed <= 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
                "Sound speed must be positive, got {}",
                sound_speed
            )));
        }

        let n_sensors = self.sensor_positions.len();
        let n_angles = angles.len();
        let mut steering_matrix = Array2::zeros([n_sensors, n_angles]);

        for (idx, &(theta, phi)) in angles.iter().enumerate() {
            let sin_theta = theta.sin();
            let direction = [sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos()];

            let phase_delays = kwavers_math::geometry::delays::plane_wave_phase_delays(
                &self.sensor_positions,
                direction,
                frequency,
                sound_speed,
            )?;

            for (i, &phase) in phase_delays.iter().enumerate() {
                steering_matrix[[i, idx]] = Complex::new(0.0, phase).exp();
            }
        }

        Ok(steering_matrix)
    }

    /// Derive processing parameters from the current array geometry.
    #[must_use]
    pub fn processing_params(&self) -> SensorProcessingParams {
        SensorProcessingParams {
            n_sensors: self.sensor_positions.len(),
            sampling_frequency: self.sampling_frequency,
            element_spacing: self.calculate_element_spacing(),
            array_aperture: self.calculate_aperture(),
        }
    }

    /// Mean spacing between adjacent elements (m).
    #[must_use]
    fn calculate_element_spacing(&self) -> f64 {
        if self.sensor_positions.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0usize;

        for i in 0..self.sensor_positions.len() - 1 {
            let pos1 = self.sensor_positions[i];
            let pos2 = self.sensor_positions[i + 1];
            total += (pos1[2] - pos2[2])
                .mul_add(
                    pos1[2] - pos2[2],
                    (pos1[1] - pos2[1]).mul_add(pos1[1] - pos2[1], (pos1[0] - pos2[0]).powi(2)),
                )
                .sqrt();
            count += 1;
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Array aperture = max_x − min_x across all sensor positions (m).
    #[must_use]
    fn calculate_aperture(&self) -> f64 {
        if self.sensor_positions.is_empty() {
            return 0.0;
        }

        let min_x = self
            .sensor_positions
            .iter()
            .fold(f64::INFINITY, |acc, p| acc.min(p[0]));
        let max_x = self
            .sensor_positions
            .iter()
            .fold(f64::NEG_INFINITY, |acc, p| acc.max(p[0]));

        max_x - min_x
    }
}
