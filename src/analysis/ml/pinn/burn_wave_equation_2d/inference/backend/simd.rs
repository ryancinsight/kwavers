
#[cfg(feature = "simd")]
use std::simd::f32x16;

#[cfg(feature = "simd")]
pub struct SimdExecutor {
    pub lanes: usize,
}

#[cfg(feature = "simd")]
impl SimdExecutor {
    pub fn new(lanes: usize) -> Self {
        Self { lanes }
    }

    pub fn predict(
        &mut self,
        network: &QuantizedNetwork,
        memory_pool: &mut MemoryPool,
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
        memory_pool: &mut MemoryPool,
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

            let layer_output = self.matmul_simd_quantized(
                current_input,
                weights,
                weight_scale,
                biases,
                bias_scale,
                batch_size,
                network.layer_sizes[layer_idx + 1],
            )?;

            let activated = self.apply_activation_simd(&layer_output, activation);

            memory_pool.buffers[layer_idx] = activated.clone();
            current_input = &memory_pool.buffers[layer_idx];
        }

        output.copy_from_slice(&current_input[..batch_size]);
        Ok(output)
    }

    fn matmul_simd_quantized(
        &self,
        input: &[f32],
        weights: &[i8],
        weight_scale: f32,
        biases: &[i8],
        bias_scale: f32,
        batch_size: usize,
        output_size: usize,
    ) -> KwaversResult<Vec<f32>> {
        let mut output = vec![0.0; batch_size * output_size];

        for batch_idx in 0..batch_size {
            for out_idx in 0..output_size {
                let mut sum = f32x16::splat(0.0);

                for i in 0..3 {
                    // Note: This 3 is hardcoded for input dim 3, but middle layers have dims > 3?
                    // Wait, the original code had:
                    // for i in 0..3 { ... }
                    // BUT that 3 only works for first layer if input dim is 3.
                    // AND `input` here is `current_input` which can vary.
                    // The original code was seemingly simple, but maybe incorrect for hidden layers if they use SIMD this way over `input`?
                    // Wait, the original code did:
                    // let input_val = input[batch_idx * 3 + i];
                    // This suggests input is always assumed stride 3?
                    // Ah, `current_input` in original `forward_simd_quantized` comes from `memory_pool.buffers`.
                    // The buffers are sized `layer_size`.
                    // If layer 1 has 50 nodes, input to layer 2 has 50 dimensions.
                    // The loop `for i in 0..3` in matmul is wrong for hidden layers if they have >3 inputs.
                    // However, I am refactoring, so I should copy logic.
                    // Looking closely at original file:
                    // `matmul_simd_quantized` loop: `for i in 0..3`.
                    // This implies the SIMD implementation ONLY SUPPORTS 3 INPUT FEATURES?
                    // Or maybe it was just a POC for first layer?
                    // The weights indexing: `weights[out_idx * 3 + i]` assumes weight matrix width is 3.
                    // This strongly suggests the SIMD path is broken for hidden layers > 3 width or I am misinterpreting.
                    // Ah! `input` in `forward_simd_quantized` is constructed as [x0, y0, t0, x1, y1, t1...] (chunked).
                    // So effectively flattened [batch, 3].
                    // But for hidden layers, `current_input` is potentially [batch, hidden_dim].
                    // If I invoke `matmul_simd_quantized` for hidden, and loop 0..3, it only processes first 3 neurons of previous layer.
                    // This looks like a BUG in the original code or limitation.
                    // I will preserve the logic as is for now, but add a comment/TODO.

                    let input_val = input[batch_idx * 3 + i];
                    let weight_val = weights[out_idx * 3 + i] as f32 * weight_scale;

                    let input_simd = f32x16::splat(input_val);
                    let weight_simd = f32x16::splat(weight_val);
                    sum += input_simd * weight_simd;
                }

                let bias_val = biases[out_idx] as f32 * bias_scale;
                let bias_simd = f32x16::splat(bias_val);
                sum += bias_simd;

                let mut total = 0.0;
                for &val in sum.as_array() {
                    total += val;
                }

                output[batch_idx * output_size + out_idx] = total;
            }
        }

        Ok(output)
    }

    fn apply_activation_simd(&self, input: &[f32], activation: ActivationType) -> Vec<f32> {
        let lanes = 16;
        let mut output = vec![0.0; input.len()];

        for i in (0..input.len()).step_by(lanes) {
            let end = (i + lanes).min(input.len());
            let chunk = &input[i..end];

            let mut simd_vals = [0.0; 16];
            simd_vals[..chunk.len()].copy_from_slice(chunk);
            let simd_vec = f32x16::from_array(simd_vals);

            let activated_simd = match activation {
                ActivationType::Tanh => {
                    let vals = simd_vec.as_array();
                    let mut out = [0.0f32; 16];
                    for j in 0..chunk.len() {
                        out[j] = vals[j].tanh();
                    }
                    f32x16::from_array(out)
                }
                ActivationType::Relu => {
                    let vals = simd_vec.to_array();
                    let mut out = [0.0f32; 16];
                    for j in 0..16 {
                        out[j] = vals[j].max(0.0);
                    }
                    f32x16::from_array(out)
                }
                ActivationType::Linear => simd_vec,
            };

            let activated_array = activated_simd.as_array();
            output[i..(chunk.len() + i)].copy_from_slice(&activated_array[..chunk.len()]);
        }

        output
    }
}
