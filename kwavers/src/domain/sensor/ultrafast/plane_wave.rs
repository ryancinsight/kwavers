//! Plane Wave Imaging and Delay Calculation
//!
//! This module implements geometric delay calculations for plane wave ultrasound
//! imaging, supporting tilted plane wave transmission and coherent compounding.
//!
//! # Mathematical Foundation
//!
//! For a plane wave with tilt angle θ:
//!
//! **Transmission delay** (element delays to create plane wave):
//! ```text
//! τ_tx(x, θ) = -x·sin(θ) / c
//! ```
//!
//! **Reception delay** (time for echo from point (x,y) to reach array):
//! ```text
//! τ_rx(x, y, θ) = (x·sin(θ) + y·cos(θ)) / c
//! ```
//!
//! **Total delay** for beamforming:
//! ```text
//! τ(x, y, θ) = τ_rx(x, y, θ) - τ_tx(x, θ)
//!            = (x·sin(θ) + y·cos(θ)) / c - (-x·sin(θ) / c)
//!            = (2x·sin(θ) + y·cos(θ)) / c
//! ```
//!
//! # Coordinate System
//!
//! - **x**: Lateral position (perpendicular to beam axis), meters
//! - **y**: Axial depth (along beam direction), meters
//! - **θ**: Tilt angle (positive = beam steered right), radians
//! - **c**: Speed of sound (m/s), typically 1540 m/s in tissue
//!
//! # References
//!
//! - Jensen, J. A., et al. (2006). "Synthetic aperture ultrasound imaging."
//!   *Ultrasonics*, 44, e5-e15. DOI: 10.1016/j.ultras.2006.07.017
//!
//! - Montaldo, G., et al. (2009). "Coherent plane-wave compounding for very high
//!   frame rate ultrasonography." *IEEE TUFFC*, 56(3), 489-506.
//!
//! - Tanter, M., & Fink, M. (2014). "Ultrafast imaging in biomedical ultrasound."
//!   *IEEE TUFFC*, 61(1), 102-119.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Plane wave imaging configuration
#[derive(Debug, Clone)]
pub struct PlaneWaveConfig {
    /// Tilt angles for coherent compounding (radians)
    pub tilt_angles: Vec<f64>,
    /// Speed of sound (m/s)
    pub sound_speed: f64,
    /// Element positions (x coordinates, meters)
    pub element_positions: Vec<f64>,
    /// F-number for apodization (optional)
    pub f_number: Option<f64>,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
}

impl Default for PlaneWaveConfig {
    fn default() -> Self {
        // Default configuration for functional ultrasound (Nouhoum et al. 2021)
        let angles_deg: Vec<f64> = (-10..=10).step_by(2).map(|a| a as f64).collect();
        let tilt_angles: Vec<f64> = angles_deg.iter().map(|&a| a * PI / 180.0).collect();

        Self {
            tilt_angles,
            sound_speed: 1540.0,           // Water/tissue
            element_positions: Vec::new(), // To be set by user
            f_number: Some(1.5),           // Typical for ultrafast imaging
            sampling_frequency: 40e6,      // 40 MHz typical
        }
    }
}

/// Plane wave imaging processor
#[derive(Debug, Clone)]
pub struct PlaneWave {
    /// Configuration
    pub config: PlaneWaveConfig,
}

impl PlaneWave {
    /// Create new plane wave processor
    pub fn new(config: PlaneWaveConfig) -> Self {
        Self { config }
    }

    /// Create with standard functional ultrasound settings
    ///
    /// Uses 11 tilted plane waves from -10° to +10° (2° steps) as in Nouhoum et al. (2021).
    pub fn functional_ultrasound(element_positions: Vec<f64>) -> Self {
        Self::new(PlaneWaveConfig {
            element_positions,
            ..PlaneWaveConfig::default()
        })
    }

    /// Compute plane wave transmission delays for array elements
    ///
    /// Returns delay for each element to create a plane wave at tilt angle θ.
    ///
    /// # Arguments
    ///
    /// * `tilt_angle` - Plane wave tilt angle (radians), positive = rightward
    ///
    /// # Returns
    ///
    /// Array of delays (seconds), one per element
    ///
    /// # Formula
    ///
    /// τ_tx(x, θ) = -x·sin(θ) / c
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pw = PlaneWave::functional_ultrasound(element_positions);
    /// let delays = pw.transmission_delays(5.0 * PI / 180.0)?; // 5° tilt
    /// ```
    pub fn transmission_delays(&self, tilt_angle: f64) -> KwaversResult<Array1<f64>> {
        if self.config.element_positions.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Element positions not set".to_string(),
            ));
        }

        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();

        let delays: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x| -x * sin_theta / c)
            .collect();

        Ok(Array1::from_vec(delays))
    }

    /// Compute reception delays for beamforming at image point (x, y)
    ///
    /// Returns delay from image point to each array element for reception beamforming.
    ///
    /// # Arguments
    ///
    /// * `x` - Lateral position (m)
    /// * `y` - Axial depth (m)
    /// * `tilt_angle` - Plane wave tilt angle (radians)
    ///
    /// # Returns
    ///
    /// Array of reception delays (seconds), one per element
    ///
    /// # Formula
    ///
    /// τ_rx(x_elem, y, θ) = ((x_elem - x)·sin(θ) + y·cos(θ)) / c
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pw = PlaneWave::functional_ultrasound(element_positions);
    /// let rx_delays = pw.reception_delays(0.005, 0.02, 0.0)?; // (5mm, 20mm, 0°)
    /// ```
    pub fn reception_delays(&self, x: f64, y: f64, tilt_angle: f64) -> KwaversResult<Array1<f64>> {
        if self.config.element_positions.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Element positions not set".to_string(),
            ));
        }

        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let cos_theta = tilt_angle.cos();

        let delays: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x_elem| ((x_elem - x) * sin_theta + y * cos_theta) / c)
            .collect();

        Ok(Array1::from_vec(delays))
    }

    /// Compute total beamforming delays (transmit + receive)
    ///
    /// Returns the combined delay for delay-and-sum beamforming at image point (x, y).
    ///
    /// # Arguments
    ///
    /// * `x` - Lateral position (m)
    /// * `y` - Axial depth (m)
    /// * `tilt_angle` - Plane wave tilt angle (radians)
    ///
    /// # Returns
    ///
    /// Array of total delays (seconds), one per element
    ///
    /// # Formula
    ///
    /// τ_total(x_elem, y, θ) = τ_rx - τ_tx
    ///                        = (2·x_elem·sin(θ) + y·cos(θ)) / c
    ///
    /// Simplifies to:
    /// ```text
    /// τ(x_elem, y, θ) = (2·x_elem·sin(θ) + y·cos(θ)) / c
    /// ```
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pw = PlaneWave::functional_ultrasound(element_positions);
    /// let delays = pw.beamforming_delays(0.0, 0.02, 5.0 * PI / 180.0)?;
    /// ```
    pub fn beamforming_delays(
        &self,
        _x: f64,
        y: f64,
        tilt_angle: f64,
    ) -> KwaversResult<Array1<f64>> {
        if self.config.element_positions.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Element positions not set".to_string(),
            ));
        }

        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let cos_theta = tilt_angle.cos();

        let delays: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x_elem| (2.0 * x_elem * sin_theta + y * cos_theta) / c)
            .collect();

        Ok(Array1::from_vec(delays))
    }

    /// Compute delay surface for entire image grid
    ///
    /// Returns 2D delay matrix [N_elements × N_pixels] for all pixels in the image.
    ///
    /// # Arguments
    ///
    /// * `x_pixels` - Lateral positions of image pixels (m)
    /// * `y_pixels` - Axial positions of image pixels (m)
    /// * `tilt_angle` - Plane wave tilt angle (radians)
    ///
    /// # Returns
    ///
    /// 2D array [N_elements × N_pixels] of delays (seconds)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let x_pixels = Array1::linspace(-0.01, 0.01, 128); // ±10 mm lateral
    /// let y_pixels = Array1::linspace(0.005, 0.04, 256); // 5-40 mm depth
    /// let delay_surface = pw.delay_surface(&x_pixels, &y_pixels, 0.0)?;
    /// ```
    pub fn delay_surface(
        &self,
        x_pixels: &Array1<f64>,
        y_pixels: &Array1<f64>,
        tilt_angle: f64,
    ) -> KwaversResult<Array2<f64>> {
        let n_elements = self.config.element_positions.len();
        let n_pixels = x_pixels.len() * y_pixels.len();

        let mut delays = Array2::zeros((n_elements, n_pixels));

        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let cos_theta = tilt_angle.cos();

        let mut pixel_idx = 0;
        for &y in y_pixels.iter() {
            for _x in x_pixels.iter() {
                for (elem_idx, &x_elem) in self.config.element_positions.iter().enumerate() {
                    let delay = (2.0 * x_elem * sin_theta + y * cos_theta) / c;
                    delays[[elem_idx, pixel_idx]] = delay;
                }
                pixel_idx += 1;
            }
        }

        Ok(delays)
    }

    /// Compute F-number dependent apodization weights
    ///
    /// Returns apodization weights based on F-number to reduce side lobes.
    ///
    /// # Arguments
    ///
    /// * `x` - Lateral position (m)
    /// * `y` - Axial depth (m)
    ///
    /// # Returns
    ///
    /// Array of apodization weights [0, 1], one per element
    ///
    /// # Formula
    ///
    /// F-number: F = depth / aperture
    /// Active aperture width: D = depth / F#
    /// Weight = 1 if |x_elem - x| < D/2, else 0 (rectangular)
    /// Or use Hann/Hamming window for smoother apodization
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let apod = pw.apodization_weights(0.0, 0.02)?; // On-axis at 20mm depth
    /// ```
    pub fn apodization_weights(&self, x: f64, y: f64) -> KwaversResult<Array1<f64>> {
        let f_number = self.config.f_number.unwrap_or(1.5);
        let aperture_width = y.abs() / f_number;
        let half_aperture = aperture_width / 2.0;

        let weights: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x_elem| {
                let distance = (x_elem - x).abs();
                if distance < half_aperture {
                    // Hann window apodization
                    let normalized_pos = distance / half_aperture; // [0, 1]
                    let angle = PI * normalized_pos; // [0, π]
                    0.5 * (1.0 + angle.cos()) // Hann window
                } else {
                    0.0
                }
            })
            .collect();

        Ok(Array1::from_vec(weights))
    }

    /// Get number of compounding angles
    pub fn num_angles(&self) -> usize {
        self.config.tilt_angles.len()
    }

    /// Get tilt angles in degrees
    pub fn angles_degrees(&self) -> Vec<f64> {
        self.config
            .tilt_angles
            .iter()
            .map(|&theta| theta * 180.0 / PI)
            .collect()
    }

    /// Estimate compounded frame rate
    ///
    /// # Arguments
    ///
    /// * `prf` - Pulse repetition frequency (Hz)
    ///
    /// # Returns
    ///
    /// Compounded frame rate (Hz) = PRF / N_angles
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let frame_rate = pw.compounded_frame_rate(5500.0); // 5500 Hz PRF
    /// // With 11 angles: 5500 / 11 = 500 Hz
    /// ```
    pub fn compounded_frame_rate(&self, prf: f64) -> f64 {
        prf / self.num_angles() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_plane_wave_transmission_delays() {
        // Create linear array: 5 elements at positions -2, -1, 0, 1, 2 mm
        let positions = vec![-0.002, -0.001, 0.0, 0.001, 0.002];
        let config = PlaneWaveConfig {
            element_positions: positions,
            sound_speed: 1540.0,
            ..Default::default()
        };
        let pw = PlaneWave::new(config);

        // Test 0° (normal incidence)
        let delays_0deg = pw.transmission_delays(0.0).unwrap();
        assert_eq!(delays_0deg.len(), 5);
        // All delays should be zero for 0° tilt
        for &delay in delays_0deg.iter() {
            assert_relative_eq!(delay, 0.0, epsilon = 1e-12);
        }

        // Test 5° tilt
        let theta = 5.0 * PI / 180.0;
        let delays_5deg = pw.transmission_delays(theta).unwrap();
        assert_eq!(delays_5deg.len(), 5);

        // Check delay progression (should be linear with position)
        let expected_delay_per_mm = -theta.sin() / 1540.0;
        assert_relative_eq!(
            delays_5deg[1] - delays_5deg[0],
            expected_delay_per_mm * 0.001,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_beamforming_delays() {
        let positions = vec![-0.001, 0.0, 0.001]; // 3 elements at -1mm, 0, +1mm
        let config = PlaneWaveConfig {
            element_positions: positions,
            sound_speed: 1540.0,
            ..Default::default()
        };
        let pw = PlaneWave::new(config);

        // Beamform to point (0mm, 20mm) with 0° tilt
        let delays = pw.beamforming_delays(0.0, 0.02, 0.0).unwrap();
        assert_eq!(delays.len(), 3);

        // For 0° tilt and on-axis point: τ = y·cos(0°)/c = 0.02/1540
        let expected = 0.02 / 1540.0;
        for &delay in delays.iter() {
            assert_relative_eq!(delay, expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_apodization_weights() {
        let positions: Vec<f64> = (0..128).map(|i| (i as f64 - 63.5) * 0.00011).collect();
        let config = PlaneWaveConfig {
            element_positions: positions,
            f_number: Some(1.5),
            ..Default::default()
        };
        let pw = PlaneWave::new(config);

        // Test apodization at 20mm depth, on-axis
        let weights = pw.apodization_weights(0.0, 0.02).unwrap();
        assert_eq!(weights.len(), 128);

        // Center element should have highest weight (1.0 for Hann)
        let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_relative_eq!(max_weight, 1.0, epsilon = 2e-4);

        // Weights should be symmetric
        assert_relative_eq!(weights[0], weights[127], epsilon = 1e-6);
        assert_relative_eq!(weights[10], weights[117], epsilon = 1e-6);
    }

    #[test]
    fn test_functional_ultrasound_config() {
        let positions: Vec<f64> = (0..128).map(|i| (i as f64 - 63.5) * 0.00011).collect();
        let pw = PlaneWave::functional_ultrasound(positions);

        // Should have 11 angles from -10° to +10°
        assert_eq!(pw.num_angles(), 11);

        let angles_deg = pw.angles_degrees();
        assert_relative_eq!(angles_deg[0], -10.0, epsilon = 0.1);
        assert_relative_eq!(angles_deg[10], 10.0, epsilon = 0.1);

        // Frame rate with PRF = 5500 Hz
        let frame_rate = pw.compounded_frame_rate(5500.0);
        assert_relative_eq!(frame_rate, 500.0, epsilon = 0.1);
    }

    #[test]
    fn test_delay_surface() {
        let positions = vec![-0.001, 0.0, 0.001];
        let config = PlaneWaveConfig {
            element_positions: positions,
            sound_speed: 1540.0,
            ..Default::default()
        };
        let pw = PlaneWave::new(config);

        let x_pixels = Array1::from_vec(vec![-0.005, 0.0, 0.005]);
        let y_pixels = Array1::from_vec(vec![0.01, 0.02]);

        let surface = pw.delay_surface(&x_pixels, &y_pixels, 0.0).unwrap();

        // Should be [3 elements × 6 pixels]
        assert_eq!(surface.dim(), (3, 6));

        // Verify a few delay values
        let y = 0.01;
        let expected_depth_delay = y / 1540.0;
        assert_relative_eq!(surface[[1, 1]], expected_depth_delay, epsilon = 1e-9);
    }
}
