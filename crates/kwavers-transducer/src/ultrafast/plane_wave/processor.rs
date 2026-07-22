//! `UltrafastPlaneWave` — plane wave processor and delay calculator.

use super::config::UltrafastPlaneWaveConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2};
use std::f64::consts::PI;

/// Plane wave imaging processor.
///
/// Computes geometric delays for plane wave ultrasound imaging and coherent compounding.
///
/// ## Mathematical Foundation
///
/// For a plane wave with tilt angle θ:
/// - Transmission delay: `τ_tx(x, θ) = -x·sin(θ) / c`
/// - Reception delay: `τ_rx(x, y, θ) = (x·sin(θ) + y·cos(θ)) / c`
/// - Total beamforming delay: `τ(x_elem, y, θ) = (2·x_elem·sin(θ) + y·cos(θ)) / c`
///
/// ## References
///
/// - Jensen, J. A., et al. (2006). Ultrasonics, 44, e5–e15.
/// - Montaldo, G., et al. (2009). IEEE TUFFC, 56(3), 489–506.
/// - Tanter, M., & Fink, M. (2014). IEEE TUFFC, 61(1), 102–119.
#[derive(Debug, Clone)]
pub struct UltrafastPlaneWave {
    /// Imaging configuration.
    pub config: UltrafastPlaneWaveConfig,
}

impl UltrafastPlaneWave {
    /// Create a new plane wave processor.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: UltrafastPlaneWaveConfig) -> Self {
        Self { config }
    }

    /// Create with standard functional ultrasound settings (11 angles, −10° to +10°).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn functional_ultrasound(element_positions: Vec<f64>) -> Self {
        Self::new(UltrafastPlaneWaveConfig {
            element_positions,
            ..UltrafastPlaneWaveConfig::default()
        })
    }

    /// Compute transmission delays: `τ_tx(x, θ) = -x·sin(θ) / c`.
    ///
    /// Returns one delay per element.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn transmission_delays(&self, tilt_angle: f64) -> KwaversResult<Array1<f64>> {
        self.require_elements()?;
        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let delays: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x| -x * sin_theta / c)
            .collect();
        Array1::from_vec([delays.len()], delays).map_err(|err| KwaversError::Shape(err.to_string()))
    }

    /// Compute reception delays: `τ_rx(x_elem, y, θ) = ((x_elem−x)·sin(θ) + y·cos(θ)) / c`.
    ///
    /// Returns one delay per element.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn reception_delays(&self, x: f64, y: f64, tilt_angle: f64) -> KwaversResult<Array1<f64>> {
        self.require_elements()?;
        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let cos_theta = tilt_angle.cos();
        let delays: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x_elem| (x_elem - x).mul_add(sin_theta, y * cos_theta) / c)
            .collect();
        Array1::from_vec([delays.len()], delays).map_err(|err| KwaversError::Shape(err.to_string()))
    }

    /// Compute total beamforming delays: `τ = (2·x_elem·sin(θ) + y·cos(θ)) / c`.
    ///
    /// Returns one delay per element.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn beamforming_delays(
        &self,
        _x: f64,
        y: f64,
        tilt_angle: f64,
    ) -> KwaversResult<Array1<f64>> {
        self.require_elements()?;
        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let cos_theta = tilt_angle.cos();
        let delays: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x_elem| (2.0 * x_elem).mul_add(sin_theta, y * cos_theta) / c)
            .collect();
        Array1::from_vec([delays.len()], delays).map_err(|err| KwaversError::Shape(err.to_string()))
    }

    /// Compute delay surface for an image grid.
    ///
    /// Returns `[N_elements × N_pixels]` delay matrix (seconds).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn delay_surface(
        &self,
        x_pixels: &Array1<f64>,
        y_pixels: &Array1<f64>,
        tilt_angle: f64,
    ) -> KwaversResult<Array2<f64>> {
        let n_elements = self.config.element_positions.len();
        let n_pixels = x_pixels.len() * y_pixels.len();
        let mut delays = Array2::zeros([n_elements, n_pixels]);

        let c = self.config.sound_speed;
        let sin_theta = tilt_angle.sin();
        let cos_theta = tilt_angle.cos();

        let mut pixel_idx = 0;
        for &y in y_pixels.iter() {
            for _ in x_pixels.iter() {
                for (elem_idx, &x_elem) in self.config.element_positions.iter().enumerate() {
                    delays[[elem_idx, pixel_idx]] =
                        (2.0 * x_elem).mul_add(sin_theta, y * cos_theta) / c;
                }
                pixel_idx += 1;
            }
        }

        Ok(delays)
    }

    /// Compute F-number dependent Hann apodization weights.
    ///
    /// Active aperture width: `D = |y| / F#`; weight is Hann-windowed within `D/2`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apodization_weights(&self, x: f64, y: f64) -> KwaversResult<Array1<f64>> {
        let f_number = self.config.f_number.unwrap_or(1.5);
        let half_aperture = y.abs() / (2.0 * f_number);

        let weights: Vec<f64> = self
            .config
            .element_positions
            .iter()
            .map(|&x_elem| {
                let distance = (x_elem - x).abs();
                if distance < half_aperture {
                    let normalized_pos = distance / half_aperture;
                    0.5 * (1.0 + (PI * normalized_pos).cos())
                } else {
                    0.0
                }
            })
            .collect();

        Array1::from_vec([weights.len()], weights)
            .map_err(|err| KwaversError::Shape(err.to_string()))
    }

    /// Number of compounding angles.
    #[must_use]
    pub fn num_angles(&self) -> usize {
        self.config.tilt_angles.len()
    }

    /// Tilt angles converted to degrees.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn angles_degrees(&self) -> Vec<f64> {
        self.config
            .tilt_angles
            .iter()
            .map(|&theta| theta.to_degrees())
            .collect()
    }

    /// Compounded frame rate: `PRF / N_angles`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn compounded_frame_rate(&self, prf: f64) -> f64 {
        prf / self.num_angles() as f64
    }

    fn require_elements(&self) -> KwaversResult<()> {
        if self.config.element_positions.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Element positions not set".to_owned(),
            ));
        }
        Ok(())
    }
}
