//! GPU-accelerated Delay-and-Sum (DAS) beamforming using Burn framework
//!
//! # Mathematical Foundation
//!
//! Delay-and-Sum beamforming focuses ultrasound signals by applying geometric delays:
//!
//! ```text
//! y[n] = Σᵢ wᵢ · xᵢ[n - τᵢ]
//! ```
//!
//! # References
//!
//! - Jensen, J.A. (1996). Field: A program for simulating ultrasound systems
//! - Montaldo, G. et al. (2009). Coherent plane-wave compounding for very high frame rate ultrasonography

use burn::tensor::{backend::Backend, Device, Tensor};
use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

/// GPU-accelerated Delay-and-Sum beamformer using Burn framework
#[derive(Debug)]
pub struct BurnDasBeamformer<B: Backend> {
    pub(super) device: Device<B>,
    pub(super) _backend: std::marker::PhantomData<B>,
}

/// Configuration for Burn-based beamforming
#[derive(Debug, Clone)]
pub struct BurnBeamformingConfig {
    pub interpolation: InterpolationMethod,
    pub batch_size: usize,
    pub async_processing: bool,
}

/// Interpolation methods for delay application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    NearestNeighbor,
    Linear,
    Cubic,
}

impl Default for BurnBeamformingConfig {
    fn default() -> Self {
        Self {
            interpolation: InterpolationMethod::NearestNeighbor,
            batch_size: 8,
            async_processing: false,
        }
    }
}

impl<B: Backend> BurnDasBeamformer<B> {
    /// Create a new Burn-based DAS beamformer
    pub fn new(device: Device<B>) -> Self {
        Self {
            device,
            _backend: std::marker::PhantomData,
        }
    }

    /// Perform delay-and-sum beamforming on RF data
    ///
    /// # Arguments
    ///
    /// * `rf_data` - RF signals [n_sensors × n_frames × n_samples]
    /// * `sensor_positions` - Sensor coordinates [n_sensors × 3]
    /// * `focal_points` - Target focal points [n_focal_points × 3]
    /// * `apodization` - Optional sensor weights [n_sensors]
    /// * `sampling_rate` - Sampling frequency (Hz)
    /// * `sound_speed` - Propagation velocity (m/s)
    pub fn beamform(
        &self,
        rf_data: &Array3<f64>,
        sensor_positions: &ndarray::Array2<f64>,
        focal_points: &ndarray::Array2<f64>,
        apodization: Option<&[f64]>,
        sampling_rate: f64,
        sound_speed: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (n_sensors, _n_frames, n_samples) = rf_data.dim();

        if sensor_positions.nrows() != n_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Sensor positions ({}) must match RF data sensors ({})",
                sensor_positions.nrows(),
                n_sensors
            )));
        }

        let rf_tensor = self.array_to_tensor_3d(rf_data)?;
        let sensor_pos_tensor = self.array_to_tensor_2d(sensor_positions)?;
        let focal_points_tensor = self.array_to_tensor_2d(focal_points)?;

        let apod_weights = if let Some(apod) = apodization {
            let apod_f32: Vec<f32> = apod.iter().map(|&x| x as f32).collect();
            Tensor::<B, 1>::from_data(apod_f32.as_slice(), &self.device).reshape([n_sensors])
        } else {
            Tensor::<B, 1>::ones([n_sensors], &self.device)
        };

        let beamformed = self.beamform_batch_tensor(
            &rf_tensor,
            &sensor_pos_tensor,
            &focal_points_tensor,
            &apod_weights,
            sampling_rate,
            sound_speed,
            n_samples,
        )?;

        self.tensor_to_array_3d(&beamformed)
    }

    /// Get the device used by this beamformer
    pub fn device(&self) -> &Device<B> {
        &self.device
    }
}

mod algorithm;
mod convert;
mod cpu;
#[cfg(test)]
mod tests;

pub use cpu::beamform_cpu;
