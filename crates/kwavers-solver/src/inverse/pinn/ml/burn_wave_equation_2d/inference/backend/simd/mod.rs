#[cfg(all(feature = "simd", feature = "nightly"))]
use crate::inverse::pinn::ml::burn_wave_equation_2d::inference::types::{
    ActivationType, BurnWave2dInferenceMemoryPool, QuantizedNetwork,
};
#[cfg(all(feature = "simd", feature = "nightly"))]
use kwavers_core::error::KwaversResult;

#[cfg(all(test, feature = "simd", feature = "nightly"))]
mod tests;

#[cfg(all(feature = "simd", feature = "nightly"))]
#[derive(Debug)]
pub struct SimdExecutor {
    pub lanes: usize,
}

#[cfg(all(feature = "simd", feature = "nightly"))]
impl SimdExecutor {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(lanes: usize) -> Self {
        Self { lanes }
    }
    /// Predict.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn predict(
        &mut self,
        network: &QuantizedNetwork,
        memory_pool: &mut BurnWave2dInferenceMemoryPool,
        x: &[f32],
        y: &[f32],
        t: &[f32],
    ) -> KwaversResult<(Vec<f32>, Vec<f32>)> {
        let batch_size = x.len();
        let mut predictions = vec![0.0; batch_size];
        let uncertainties = vec![0.01; batch_size];

        let chunk_size = self.lanes;

        for chunk_start in (0..batch_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(batch_size);

            let x_chunk = &x[chunk_start..chunk_end];
            let y_chunk = &y[chunk_start..chunk_end];
            let t_chunk = &t[chunk_start..chunk_end];

            let output_chunk =
                self.forward_simd_quantized(network, memory_pool, x_chunk, y_chunk, t_chunk)?;

            for (i, &pred) in output_chunk.iter().enumerate() {
                predictions[chunk_start + i] = pred;
            }
        }

        Ok((predictions, uncertainties))
    }

    fn forward_simd_quantized(
        &mut self,
        network: &QuantizedNetwork,
        memory_pool: &mut BurnWave2dInferenceMemoryPool,
        x: &[f32],
        y: &[f32],
        t: &[f32],
    ) -> KwaversResult<Vec<f32>> {
        let batch_size = x.len();
        let mut output = vec![0.0; batch_size];

        let mut input = Vec::with_capacity(batch_size * 3);
        for i in 0..batch_size {
            input.push(x[i]);
            input.push(y[i]);
            input.push(t[i]);
        }

        let mut current_input = &input;

        for layer_idx in 0..network.weights.len() {
            let weights = &network.weights[layer_idx];
            let biases = &network.biases[layer_idx];
            let weight_scale = network.weight_scales[layer_idx];
            let bias_scale = network.bias_scales[layer_idx];
            let activation = network.activations[layer_idx];

            let input_size = if layer_idx == 0 {
                3 // First layer: (x, y, t)
            } else {
                network.layer_sizes[layer_idx]
            };

            let layer_output = self.matmul_simd_quantized(
                current_input,
                weights,
                weight_scale,
                biases,
                bias_scale,
                batch_size,
                input_size,
                network.layer_sizes[layer_idx + 1],
            )?;

            let activated = self.apply_activation_simd(&layer_output, activation);

            memory_pool.buffers[layer_idx] = activated.clone();
            current_input = &memory_pool.buffers[layer_idx];
        }

        output.copy_from_slice(&current_input[..batch_size]);
        Ok(output)
    }

    /// Matmul simd quantized.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    // Independent buffers, scales, and dimensions for a quantized matmul with no
    // cohesive sub-grouping; bundling would not clarify the call site.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_simd_quantized(
        &self,
        input: &[f32],
        weights: &[i8],
        weight_scale: f32,
        biases: &[i8],
        bias_scale: f32,
        batch_size: usize,
        input_size: usize,
        output_size: usize,
    ) -> KwaversResult<Vec<f32>> {
        let mut output = vec![0.0; batch_size * output_size];

        for batch_idx in 0..batch_size {
            for out_idx in 0..output_size {
                for i in 0..input_size {
                    let input_val = input[batch_idx * input_size + i];
                    let weight_val = weights[out_idx * input_size + i] as f32 * weight_scale;
                    output[batch_idx * output_size + out_idx] += input_val * weight_val;
                }

                let bias_val = biases[out_idx] as f32 * bias_scale;
                output[batch_idx * output_size + out_idx] += bias_val;
            }
        }

        Ok(output)
    }

    fn apply_activation_simd(&self, input: &[f32], activation: ActivationType) -> Vec<f32> {
        input
            .iter()
            .map(|&val| match activation {
                ActivationType::Tanh => val.tanh(),
                ActivationType::Relu => val.max(0.0),
                ActivationType::Linear => val,
            })
            .collect()
    }
}
