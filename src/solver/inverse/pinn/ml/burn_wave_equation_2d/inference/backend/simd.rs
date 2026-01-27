#[cfg(all(feature = "simd", feature = "nightly"))]
use crate::core::error::KwaversResult;
#[cfg(all(feature = "simd", feature = "nightly"))]
use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::inference::types::{
    ActivationType, MemoryPool, QuantizedNetwork,
};
#[cfg(all(feature = "simd", feature = "nightly"))]
#[derive(Debug)]
pub struct SimdExecutor {
    pub lanes: usize,
}

#[cfg(all(feature = "simd", feature = "nightly"))]
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

    fn matmul_simd_quantized(
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

#[cfg(all(test, feature = "simd", feature = "nightly"))]
mod tests {
    use super::*;
    use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::inference::types::{
        ActivationType, MemoryPool, QuantizedNetwork,
    };

    /// Reference scalar implementation for validation
    fn matmul_scalar_quantized(
        input: &[f32],
        weights: &[i8],
        weight_scale: f32,
        biases: &[i8],
        bias_scale: f32,
        batch_size: usize,
        input_size: usize,
        output_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_size * output_size];

        for batch_idx in 0..batch_size {
            for out_idx in 0..output_size {
                let mut sum = 0.0;

                for i in 0..input_size {
                    let input_val = input[batch_idx * input_size + i];
                    let weight_val = weights[out_idx * input_size + i] as f32 * weight_scale;
                    sum += input_val * weight_val;
                }

                let bias_val = biases[out_idx] as f32 * bias_scale;
                sum += bias_val;

                output[batch_idx * output_size + out_idx] = sum;
            }
        }

        output
    }

    /// Test SIMD matmul with input_size=3, output_size=3
    #[test]
    fn test_matmul_simd_3x3() {
        let executor = SimdExecutor::new(16);

        let batch_size = 2;
        let input_size = 3;
        let output_size = 3;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let weight_scale = 0.1;
        let biases = vec![10, 20, 30];
        let bias_scale = 0.01;

        let simd_result = executor
            .matmul_simd_quantized(
                &input,
                &weights,
                weight_scale,
                &biases,
                bias_scale,
                batch_size,
                input_size,
                output_size,
            )
            .unwrap();

        let scalar_result = matmul_scalar_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        );

        assert_eq!(simd_result.len(), scalar_result.len());
        for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-5,
                "SIMD output {} != scalar output {}",
                simd_val,
                scalar_val
            );
        }
    }

    /// Test SIMD matmul with input_size=3, output_size=8 (hidden layer)
    #[test]
    fn test_matmul_simd_3x8() {
        let executor = SimdExecutor::new(16);

        let batch_size = 2;
        let input_size = 3;
        let output_size = 8;

        let input = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
        let weights: Vec<i8> = (0..24).map(|i| (i % 127) as i8).collect();
        let weight_scale = 0.05;
        let biases: Vec<i8> = (0..8).map(|i| (i * 5) as i8).collect();
        let bias_scale = 0.02;

        let simd_result = executor
            .matmul_simd_quantized(
                &input,
                &weights,
                weight_scale,
                &biases,
                bias_scale,
                batch_size,
                input_size,
                output_size,
            )
            .unwrap();

        let scalar_result = matmul_scalar_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        );

        assert_eq!(simd_result.len(), scalar_result.len());
        for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-5,
                "SIMD output {} != scalar output {}",
                simd_val,
                scalar_val
            );
        }
    }

    /// Test SIMD matmul with input_size=16, output_size=16 (larger hidden layer)
    #[test]
    fn test_matmul_simd_16x16() {
        let executor = SimdExecutor::new(16);

        let batch_size = 4;
        let input_size = 16;
        let output_size = 16;

        let input: Vec<f32> = (0..batch_size * input_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let weights: Vec<i8> = (0..input_size * output_size)
            .map(|i| ((i * 7) % 127) as i8)
            .collect();
        let weight_scale = 0.03;
        let biases: Vec<i8> = (0..output_size).map(|i| (i * 3) as i8).collect();
        let bias_scale = 0.015;

        let simd_result = executor
            .matmul_simd_quantized(
                &input,
                &weights,
                weight_scale,
                &biases,
                bias_scale,
                batch_size,
                input_size,
                output_size,
            )
            .unwrap();

        let scalar_result = matmul_scalar_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        );

        assert_eq!(simd_result.len(), scalar_result.len());
        for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-4,
                "SIMD output {} != scalar output {}",
                simd_val,
                scalar_val
            );
        }
    }

    /// Test SIMD matmul with input_size=32, output_size=1 (output layer from large hidden)
    #[test]
    fn test_matmul_simd_32x1() {
        let executor = SimdExecutor::new(16);

        let batch_size = 8;
        let input_size = 32;
        let output_size = 1;

        let input: Vec<f32> = (0..batch_size * input_size)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let weights: Vec<i8> = (0..input_size).map(|i| ((i * 11) % 127) as i8).collect();
        let weight_scale = 0.04;
        let biases = vec![42];
        let bias_scale = 0.01;

        let simd_result = executor
            .matmul_simd_quantized(
                &input,
                &weights,
                weight_scale,
                &biases,
                bias_scale,
                batch_size,
                input_size,
                output_size,
            )
            .unwrap();

        let scalar_result = matmul_scalar_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        );

        assert_eq!(simd_result.len(), scalar_result.len());
        for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-4,
                "SIMD output {} != scalar output {}",
                simd_val,
                scalar_val
            );
        }
    }

    /// Integration test: full forward pass with multi-layer network
    #[test]
    fn test_forward_simd_multilayer() {
        let mut executor = SimdExecutor::new(16);

        // Network: 3 → 8 → 4 → 1
        let layer_sizes = vec![3, 8, 4, 1];
        let num_layers = layer_sizes.len() - 1;

        // Create quantized network
        let mut weights = Vec::new();
        let mut weight_scales = Vec::new();
        let mut biases = Vec::new();
        let mut bias_scales = Vec::new();
        let mut activations = Vec::new();

        for i in 0..num_layers {
            let in_size = layer_sizes[i];
            let out_size = layer_sizes[i + 1];
            let w: Vec<i8> = (0..in_size * out_size)
                .map(|j| ((j * 13 + i * 7) % 127) as i8)
                .collect();
            let b: Vec<i8> = (0..out_size).map(|j| ((j * 5) % 50) as i8).collect();

            weights.push(w);
            weight_scales.push(0.05);
            biases.push(b);
            bias_scales.push(0.01);
            activations.push(if i < num_layers - 1 {
                ActivationType::Tanh
            } else {
                ActivationType::Linear
            });
        }

        let network = QuantizedNetwork {
            weights,
            weight_scales,
            biases,
            bias_scales,
            layer_sizes: layer_sizes.clone(),
            activations,
        };

        let mut memory_pool = MemoryPool {
            buffers: vec![vec![0.0; 256]; num_layers],
            _buffer_sizes: layer_sizes[1..].to_vec(),
        };

        // Test inputs
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let t = vec![0.5, 1.0];

        let result = executor.predict(&network, &mut memory_pool, &x, &y, &t);

        assert!(result.is_ok(), "Forward pass should succeed");
        let (predictions, uncertainties) = result.unwrap();
        assert_eq!(predictions.len(), 2);
        assert_eq!(uncertainties.len(), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }
}
