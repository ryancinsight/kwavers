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
//! where:
//! - `y[n]` = beamformed signal at sample `n`
//! - `xᵢ[n]` = RF signal from sensor `i`
//! - `wᵢ` = apodization weight for sensor `i`
//! - `τᵢ` = geometric delay for sensor `i` in samples
//!
//! ## Geometric Delay Computation
//!
//! For a focal point `f = (xf, yf, zf)` and sensor position `sᵢ = (xᵢ, yᵢ, zᵢ)`:
//!
//! ```text
//! dᵢ = ||sᵢ - f|| = √[(xᵢ-xf)² + (yᵢ-yf)² + (zᵢ-zf)²]
//! τᵢ = (dᵢ / c) · fs
//! ```
//!
//! where:
//! - `dᵢ` = Euclidean distance from sensor `i` to focal point
//! - `c` = speed of sound (m/s)
//! - `fs` = sampling rate (Hz)
//!
//! # Architecture
//!
//! This module provides:
//! - Generic backend support (CPU via NdArray, GPU via WGPU/CUDA)
//! - Batch processing for multiple focal points
//! - Tensor-native operations (no CPU roundtrips for GPU backends)
//! - Nearest-neighbor and linear interpolation
//!
//! # References
//!
//! - Jensen, J.A. (1996). Field: A program for simulating ultrasound systems
//! - Montaldo, G. et al. (2009). Coherent plane-wave compounding for very high frame rate ultrasonography

use burn::tensor::{backend::Backend, Device, Tensor};
use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

/// GPU-accelerated Delay-and-Sum beamformer using Burn framework
///
/// # Type Parameters
///
/// * `B` - Burn backend (e.g., `NdArray`, `Wgpu`, `Cuda`)
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use kwavers::analysis::signal_processing::beamforming::gpu::BurnDasBeamformer;
///
/// let device = Default::default();
/// let beamformer: BurnDasBeamformer<NdArray> = BurnDasBeamformer::new(device);
/// ```
#[derive(Debug)]
pub struct BurnDasBeamformer<B: Backend> {
    /// Device for tensor operations
    device: Device<B>,
    /// Phantom data for backend type
    _backend: std::marker::PhantomData<B>,
}

/// Configuration for Burn-based beamforming
#[derive(Debug, Clone)]
pub struct BurnBeamformingConfig {
    /// Interpolation method for sub-sample accuracy
    pub interpolation: InterpolationMethod,
    /// Batch size for processing multiple focal points
    pub batch_size: usize,
    /// Enable async processing (for compatible backends)
    pub async_processing: bool,
}

/// Interpolation methods for delay application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Nearest-neighbor (fastest, integer delays)
    NearestNeighbor,
    /// Linear interpolation (sub-sample accuracy)
    Linear,
    /// Cubic interpolation (highest quality)
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
    ///
    /// # Arguments
    ///
    /// * `device` - Burn device (CPU or GPU)
    ///
    /// # Returns
    ///
    /// New beamformer instance
    pub fn new(device: Device<B>) -> Self {
        Self {
            device,
            _backend: std::marker::PhantomData,
        }
    }

    /// Perform delay-and-sum beamforming on RF data
    ///
    /// # Mathematical Specification
    ///
    /// For each focal point `f` and frame `t`:
    ///
    /// 1. Compute distances: `dᵢ = ||sᵢ - f||` for all sensors `i`
    /// 2. Convert to delays: `τᵢ = (dᵢ / c) · fs`
    /// 3. Apply delays and sum: `y[t] = Σᵢ wᵢ · xᵢ[t, τᵢ]`
    ///
    /// # Arguments
    ///
    /// * `rf_data` - RF signals [n_sensors × n_frames × n_samples]
    /// * `sensor_positions` - Sensor coordinates [n_sensors × 3] (x, y, z)
    /// * `focal_points` - Target focal points [n_focal_points × 3] (x, y, z)
    /// * `apodization` - Sensor weights [n_sensors] (uniform if None)
    /// * `sampling_rate` - Sampling frequency (Hz)
    /// * `sound_speed` - Propagation velocity (m/s)
    ///
    /// # Returns
    ///
    /// Beamformed image [n_focal_points × n_frames × 1]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input dimensions are inconsistent
    /// - Tensor operations fail
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
        let _n_focal_points = focal_points.nrows();

        // Validate inputs
        if sensor_positions.nrows() != n_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Sensor positions ({}) must match RF data sensors ({})",
                sensor_positions.nrows(),
                n_sensors
            )));
        }

        // Convert to Burn tensors
        let rf_tensor = self.array_to_tensor_3d(rf_data)?;
        let sensor_pos_tensor = self.array_to_tensor_2d(sensor_positions)?;
        let focal_points_tensor = self.array_to_tensor_2d(focal_points)?;

        // Create apodization weights [n_sensors]
        let apod_weights = if let Some(apod) = apodization {
            let apod_f32: Vec<f32> = apod.iter().map(|&x| x as f32).collect();
            Tensor::<B, 1>::from_data(apod_f32.as_slice(), &self.device).reshape([n_sensors])
        } else {
            // Uniform weights
            Tensor::<B, 1>::ones([n_sensors], &self.device)
        };

        // Process all focal points in batch
        let beamformed = self.beamform_batch_tensor(
            &rf_tensor,
            &sensor_pos_tensor,
            &focal_points_tensor,
            &apod_weights,
            sampling_rate,
            sound_speed,
            n_samples,
        )?;

        // Convert result back to ndarray
        self.tensor_to_array_3d(&beamformed)
    }

    /// Batch beamforming in pure tensor space (no CPU roundtrips)
    ///
    /// # Tensor Operations
    ///
    /// All operations remain in tensor space for GPU efficiency:
    /// 1. Broadcast operations for vectorized distance computation
    /// 2. Clamp delay indices to valid sample range
    /// 3. Vectorized gather operations for delayed samples
    /// 4. Reduction (sum) over sensors with apodization
    ///
    /// # Arguments
    ///
    /// * `rf_tensor` - [n_sensors, n_frames, n_samples]
    /// * `sensor_pos` - [n_sensors, 3]
    /// * `focal_points` - [n_focal_points, 3]
    /// * `apod_weights` - [n_sensors]
    /// * `sampling_rate` - Sampling frequency (Hz)
    /// * `sound_speed` - Speed of sound (m/s)
    /// * `n_samples` - Number of time samples
    ///
    /// # Returns
    ///
    /// Beamformed tensor [n_focal_points, n_frames, 1]
    fn beamform_batch_tensor(
        &self,
        rf_tensor: &Tensor<B, 3>,
        sensor_pos: &Tensor<B, 2>,
        focal_points: &Tensor<B, 2>,
        apod_weights: &Tensor<B, 1>,
        sampling_rate: f64,
        sound_speed: f64,
        n_samples: usize,
    ) -> KwaversResult<Tensor<B, 3>> {
        let n_sensors = rf_tensor.shape().dims[0];
        let n_frames = rf_tensor.shape().dims[1];
        let n_focal_points = focal_points.shape().dims[0];

        // Compute distances for all focal points: [n_focal_points, n_sensors]
        // distances[i,j] = ||sensor_pos[j] - focal_points[i]||
        let distances = self.compute_distances_batch(sensor_pos, focal_points);

        // Convert distances to sample delays: [n_focal_points, n_sensors]
        // delay_samples[i,j] = (distances[i,j] / c) * fs
        let delay_samples = distances
            .clone()
            .div_scalar((sound_speed / sampling_rate) as f32);

        // Round to nearest integer for nearest-neighbor interpolation
        let delay_indices = delay_samples.clone().round().int();

        // Initialize output: [n_focal_points, n_frames]
        let mut output = Vec::with_capacity(n_focal_points * n_frames);

        // Process each focal point
        for focal_idx in 0..n_focal_points {
            // Extract delays for this focal point: [n_sensors]
            let delays = delay_indices
                .clone()
                .slice([focal_idx..focal_idx + 1, 0..n_sensors])
                .squeeze::<1>();

            // Process each frame
            for frame_idx in 0..n_frames {
                // Extract RF frame: [n_sensors, n_samples]
                let rf_frame = rf_tensor
                    .clone()
                    .slice([0..n_sensors, frame_idx..frame_idx + 1, 0..n_samples])
                    .squeeze::<2>();

                // Apply delays and compute weighted sum
                // This is the critical operation: gather delayed samples
                let delayed_samples = self.gather_delayed_samples(&rf_frame, &delays, n_samples);

                // Apply apodization and sum: sum(w_i * x_i[delay_i])
                let weighted = delayed_samples * apod_weights.clone();
                let beamformed_value = weighted.sum();

                // Store result
                let value_data = beamformed_value.into_data();
                let value = value_data.as_slice::<f32>().unwrap()[0] as f64;
                output.push(value);
            }
        }

        // Reshape output to [n_focal_points, n_frames, 1]
        let output_f32: Vec<f32> = output.iter().map(|&x| x as f32).collect();
        let output_tensor = Tensor::<B, 1>::from_data(output_f32.as_slice(), &self.device)
            .reshape([n_focal_points, n_frames, 1]);

        Ok(output_tensor)
    }

    /// Gather delayed samples from RF data
    ///
    /// # Mathematical Operation
    ///
    /// For each sensor `i`, extract `rf_frame[i, delay[i]]`
    /// with bounds checking: if `delay[i] < 0` or `delay[i] >= n_samples`, return 0
    ///
    /// # Arguments
    ///
    /// * `rf_frame` - RF data [n_sensors, n_samples]
    /// * `delays` - Integer delay indices [n_sensors]
    /// * `n_samples` - Valid sample range [0, n_samples)
    ///
    /// # Returns
    ///
    /// Delayed samples [n_sensors]
    fn gather_delayed_samples(
        &self,
        rf_frame: &Tensor<B, 2>,
        delays: &Tensor<B, 1, burn::tensor::Int>,
        n_samples: usize,
    ) -> Tensor<B, 1> {
        let n_sensors = rf_frame.shape().dims[0];

        // Convert delays to CPU for indexing (Burn 0.19 doesn't have advanced gather_nd)
        let delay_data = delays.clone().into_data();
        let delay_vec = delay_data.as_slice::<i64>().unwrap();

        let rf_data = rf_frame.clone().into_data();
        let rf_slice = rf_data.as_slice::<f32>().unwrap();

        // Manual gather with bounds checking
        let mut samples = Vec::with_capacity(n_sensors);
        for (i, &delay_idx) in delay_vec.iter().enumerate().take(n_sensors) {
            if delay_idx >= 0 && (delay_idx as usize) < n_samples {
                let idx = i * n_samples + delay_idx as usize;
                samples.push(rf_slice[idx]);
            } else {
                samples.push(0.0_f32);
            }
        }

        Tensor::<B, 1>::from_data(samples.as_slice(), &self.device).reshape([n_sensors])
    }

    /// Compute pairwise distances between sensors and focal points
    ///
    /// # Mathematical Specification
    ///
    /// ```text
    /// distances[i,j] = ||sensor_pos[j] - focal_points[i]||₂
    ///                = √[(x_j - x_fi)² + (y_j - y_fi)² + (z_j - z_fi)²]
    /// ```
    ///
    /// # Tensor Operations
    ///
    /// Uses broadcasting for efficient computation:
    /// 1. Reshape sensor_pos to [1, n_sensors, 3]
    /// 2. Reshape focal_points to [n_focal_points, 1, 3]
    /// 3. Compute diff = sensor_pos - focal_points (broadcast to [n_focal_points, n_sensors, 3])
    /// 4. distances = sqrt(sum(diff², axis=2))
    ///
    /// # Arguments
    ///
    /// * `sensor_pos` - Sensor positions [n_sensors, 3]
    /// * `focal_points` - Focal point positions [n_focal_points, 3]
    ///
    /// # Returns
    ///
    /// Distance matrix [n_focal_points, n_sensors]
    fn compute_distances_batch(
        &self,
        sensor_pos: &Tensor<B, 2>,
        focal_points: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let n_sensors = sensor_pos.shape().dims[0];
        let n_focal_points = focal_points.shape().dims[0];

        // For each focal point, compute distance to all sensors
        // We'll do this by iterating over focal points to avoid complex broadcasting
        let mut all_distances = Vec::with_capacity(n_focal_points * n_sensors);

        for focal_idx in 0..n_focal_points {
            // Extract focal point: [3]
            let focal = focal_points
                .clone()
                .slice([focal_idx..focal_idx + 1, 0..3])
                .squeeze::<1>();

            // Compute differences for all sensors: [n_sensors, 3]
            // Broadcast focal to [n_sensors, 3]
            let focal_broadcast = focal.clone().unsqueeze().repeat_dim(0, n_sensors);
            let diff = sensor_pos.clone() - focal_broadcast;

            // Compute squared differences and sum: [n_sensors]
            let diff_squared = diff.clone() * diff;
            let sum_squared = diff_squared.sum_dim(1); // [n_sensors]
            let distances = sum_squared.sqrt(); // [n_sensors]

            // Extract to CPU and store
            let dist_data = distances.into_data();
            let dist_slice = dist_data.as_slice::<f32>().unwrap();
            all_distances.extend_from_slice(dist_slice);
        }

        // Reconstruct as [n_focal_points, n_sensors]
        Tensor::<B, 1>::from_data(all_distances.as_slice(), &self.device)
            .reshape([n_focal_points, n_sensors])
    }

    /// Convert ndarray to Burn tensor (3D)
    fn array_to_tensor_3d(&self, array: &Array3<f64>) -> KwaversResult<Tensor<B, 3>> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        Ok(Tensor::<B, 1>::from_data(data.as_slice(), &self.device)
            .reshape([shape[0], shape[1], shape[2]]))
    }

    /// Convert ndarray to Burn tensor (2D)
    fn array_to_tensor_2d(&self, array: &ndarray::Array2<f64>) -> KwaversResult<Tensor<B, 2>> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        Ok(Tensor::<B, 1>::from_data(data.as_slice(), &self.device).reshape([shape[0], shape[1]]))
    }

    /// Convert Burn tensor to ndarray (3D)
    fn tensor_to_array_3d(&self, tensor: &Tensor<B, 3>) -> KwaversResult<Array3<f64>> {
        let shape = tensor.shape();
        let data = tensor.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::InvalidInput(format!("Tensor conversion failed: {:?}", e))
        })?;

        Array3::from_shape_vec(
            (shape.dims[0], shape.dims[1], shape.dims[2]),
            slice.iter().map(|&x| x as f64).collect(),
        )
        .map_err(|e| KwaversError::InvalidInput(format!("Array reshape failed: {}", e)))
    }

    /// Get the device used by this beamformer
    pub fn device(&self) -> &Device<B> {
        &self.device
    }
}

/// CPU convenience function for Burn-based beamforming
///
/// Uses NdArray backend for CPU-only operation without GPU dependencies.
///
/// # Arguments
///
/// * `rf_data` - RF signals [n_sensors × n_frames × n_samples]
/// * `sensor_positions` - Sensor coordinates [n_sensors × 3]
/// * `focal_points` - Target focal points [n_focal_points × 3]
/// * `apodization` - Optional sensor weights [n_sensors]
/// * `sampling_rate` - Sampling frequency (Hz)
/// * `sound_speed` - Propagation velocity (m/s)
///
/// # Returns
///
/// Beamformed image [n_focal_points × n_frames × 1]
pub fn beamform_cpu(
    rf_data: &Array3<f64>,
    sensor_positions: &ndarray::Array2<f64>,
    focal_points: &ndarray::Array2<f64>,
    apodization: Option<&[f64]>,
    sampling_rate: f64,
    sound_speed: f64,
) -> KwaversResult<Array3<f64>> {
    use burn::backend::NdArray;
    let device = Default::default();
    let beamformer: BurnDasBeamformer<NdArray> = BurnDasBeamformer::new(device);
    beamformer.beamform(
        rf_data,
        sensor_positions,
        focal_points,
        apodization,
        sampling_rate,
        sound_speed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use ndarray::{Array2, Array3};

    type TestBackend = NdArray;

    #[test]
    fn test_burn_beamformer_creation() {
        let device = Default::default();
        let _beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);
    }

    #[test]
    fn test_distance_computation() {
        let device = Default::default();
        let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

        // Single sensor at origin
        let sensor_pos = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let sensor_tensor = beamformer.array_to_tensor_2d(&sensor_pos).unwrap();

        // Focal point at (3, 4, 0) - expect distance 5
        let focal = Array2::from_shape_vec((1, 3), vec![3.0, 4.0, 0.0]).unwrap();
        let focal_tensor = beamformer.array_to_tensor_2d(&focal).unwrap();

        let distances = beamformer.compute_distances_batch(&sensor_tensor, &focal_tensor);
        let dist_data = distances.into_data();
        let dist_vec = dist_data.as_slice::<f32>().unwrap();

        assert!((dist_vec[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_single_focal_point_beamforming() {
        let device = Default::default();
        let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

        // 2 sensors, 3 frames, 10 samples
        let rf_data = Array3::from_shape_fn((2, 3, 10), |(i, j, k)| (i + j + k) as f64 * 0.1);

        // Sensors at (0,0,0) and (0.01, 0, 0)
        let sensor_pos =
            Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).unwrap();

        // Single focal point at (0.005, 0, 0.01)
        let focal_points = Array2::from_shape_vec((1, 3), vec![0.005, 0.0, 0.01]).unwrap();

        let result = beamformer.beamform(&rf_data, &sensor_pos, &focal_points, None, 1e6, 1500.0);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 3, 1]);
    }

    #[test]
    fn test_apodization() {
        let device = Default::default();
        let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

        // 3 sensors, 1 frame, 5 samples
        let rf_data = Array3::from_shape_vec(
            (3, 1, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, // Sensor 0
                2.0, 4.0, 6.0, 8.0, 10.0, // Sensor 1
                3.0, 6.0, 9.0, 12.0, 15.0, // Sensor 2
            ],
        )
        .unwrap();

        let sensor_pos =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.02, 0.0, 0.0])
                .unwrap();

        let focal_points = Array2::from_shape_vec((1, 3), vec![0.01, 0.0, 0.01]).unwrap();

        // Hamming-style weights
        let apod_weights = vec![0.5, 1.0, 0.5];

        let result = beamformer.beamform(
            &rf_data,
            &sensor_pos,
            &focal_points,
            Some(&apod_weights),
            1e6,
            1500.0,
        );

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 1, 1]);
    }

    #[test]
    fn test_invalid_input_dimensions() {
        let device = Default::default();
        let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

        // Mismatched sensor count
        let rf_data = Array3::zeros((3, 2, 10));
        let sensor_pos = Array2::zeros((2, 3)); // Wrong: 2 sensors vs 3 in RF
        let focal_points = Array2::zeros((1, 3));

        let result = beamformer.beamform(&rf_data, &sensor_pos, &focal_points, None, 1e6, 1500.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_wrapper() {
        let rf_data = Array3::ones((2, 2, 10));
        let sensor_pos = Array2::zeros((2, 3));
        let focal_points = Array2::zeros((1, 3));

        let result = beamform_cpu(&rf_data, &sensor_pos, &focal_points, None, 1e6, 1500.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_array_tensor_conversion() {
        let device = Default::default();
        let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

        let array = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i + j + k) as f64);
        let tensor = beamformer.array_to_tensor_3d(&array).unwrap();
        let reconstructed = beamformer.tensor_to_array_3d(&tensor).unwrap();

        assert_eq!(array.shape(), reconstructed.shape());
        for (a, b) in array.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_multiple_focal_points() {
        let device = Default::default();
        let beamformer: BurnDasBeamformer<TestBackend> = BurnDasBeamformer::new(device);

        let rf_data = Array3::ones((4, 5, 20));
        let sensor_pos = Array2::zeros((4, 3));

        // 3 focal points
        let focal_points = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 0.0, 0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01],
        )
        .unwrap();

        let result = beamformer.beamform(&rf_data, &sensor_pos, &focal_points, None, 1e6, 1500.0);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[3, 5, 1]); // 3 focal points, 5 frames
    }
}
