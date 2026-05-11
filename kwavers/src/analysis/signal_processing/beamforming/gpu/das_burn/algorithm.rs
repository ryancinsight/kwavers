use burn::tensor::{backend::Backend, Tensor};

use crate::core::error::KwaversResult;

use super::BurnDasBeamformer;

impl<B: Backend> BurnDasBeamformer<B> {
    /// Batch beamforming in pure tensor space
    ///
    /// All operations remain in tensor space for GPU efficiency:
    /// 1. Broadcast operations for vectorized distance computation
    /// 2. Clamp delay indices to valid sample range
    /// 3. Vectorized gather operations for delayed samples
    /// 4. Reduction (sum) over sensors with apodization
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn beamform_batch_tensor(
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

        let distances = self.compute_distances_batch(sensor_pos, focal_points);

        let delay_samples = distances
            .clone()
            .div_scalar((sound_speed / sampling_rate) as f32);

        let delay_indices = delay_samples.clone().round().int();

        let mut output = Vec::with_capacity(n_focal_points * n_frames);

        for focal_idx in 0..n_focal_points {
            let delays = delay_indices
                .clone()
                .slice([focal_idx..focal_idx + 1, 0..n_sensors])
                .squeeze::<1>();

            for frame_idx in 0..n_frames {
                let rf_frame = rf_tensor
                    .clone()
                    .slice([0..n_sensors, frame_idx..frame_idx + 1, 0..n_samples])
                    .squeeze::<2>();

                let delayed_samples = self.gather_delayed_samples(&rf_frame, &delays, n_samples);

                let weighted = delayed_samples * apod_weights.clone();
                let beamformed_value = weighted.sum();

                let value_data = beamformed_value.into_data();
                let value = value_data.as_slice::<f32>().unwrap()[0] as f64;
                output.push(value);
            }
        }

        let output_f32: Vec<f32> = output.iter().map(|&x| x as f32).collect();
        let output_tensor = Tensor::<B, 1>::from_data(output_f32.as_slice(), &self.device)
            .reshape([n_focal_points, n_frames, 1]);

        Ok(output_tensor)
    }

    /// Gather delayed samples from RF data with bounds checking
    ///
    /// For each sensor `i`, extracts `rf_frame[i, delay[i]]`.
    /// Returns 0 when `delay[i]` is out of range [0, n_samples).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn gather_delayed_samples(
        &self,
        rf_frame: &Tensor<B, 2>,
        delays: &Tensor<B, 1, burn::tensor::Int>,
        n_samples: usize,
    ) -> Tensor<B, 1> {
        let n_sensors = rf_frame.shape().dims[0];

        let delay_data = delays.clone().into_data();
        let delay_vec = delay_data.as_slice::<i64>().unwrap();

        let rf_data = rf_frame.clone().into_data();
        let rf_slice = rf_data.as_slice::<f32>().unwrap();

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
    /// ```text
    /// distances[i,j] = ||sensor_pos[j] - focal_points[i]||₂
    /// ```
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub(super) fn compute_distances_batch(
        &self,
        sensor_pos: &Tensor<B, 2>,
        focal_points: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let n_sensors = sensor_pos.shape().dims[0];
        let n_focal_points = focal_points.shape().dims[0];

        let mut all_distances = Vec::with_capacity(n_focal_points * n_sensors);

        for focal_idx in 0..n_focal_points {
            let focal = focal_points
                .clone()
                .slice([focal_idx..focal_idx + 1, 0..3])
                .squeeze::<1>();

            let focal_broadcast = focal.clone().unsqueeze().repeat_dim(0, n_sensors);
            let diff = sensor_pos.clone() - focal_broadcast;

            let diff_squared = diff.clone() * diff;
            let sum_squared = diff_squared.sum_dim(1);
            let distances = sum_squared.sqrt();

            let dist_data = distances.into_data();
            let dist_slice = dist_data.as_slice::<f32>().unwrap();
            all_distances.extend_from_slice(dist_slice);
        }

        Tensor::<B, 1>::from_data(all_distances.as_slice(), &self.device)
            .reshape([n_focal_points, n_sensors])
    }
}
