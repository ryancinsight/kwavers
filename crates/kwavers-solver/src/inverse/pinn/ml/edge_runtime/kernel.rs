use super::{Architecture, DataType, EdgeRuntime, ExecutionKernel};
use kwavers_core::error::{KwaversError, KwaversResult};

impl EdgeRuntime {
    /// Quantize input for kernel.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn quantize_input_for_kernel(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        match self.hardware_caps.architecture {
            Architecture::ARM | Architecture::ARM64 => self.neon_quantize(input, kernel),
            Architecture::RISCV => self.riscv_quantize(input, kernel),
            _ => self.software_quantize(input, kernel),
        }
    }
    /// Neon quantize.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn neon_quantize(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        let _ = kernel;
        Ok(input.iter().map(|&x| x.clamp(-1.0, 1.0)).collect())
    }
    /// Riscv quantize.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn riscv_quantize(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        let _ = kernel;
        Ok(input.iter().map(|&x| x.clamp(-1.0, 1.0)).collect())
    }
    /// Software quantize.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn software_quantize(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        match kernel.io_spec.input_dtype {
            DataType::Float32 => Ok(input.to_vec()),
            DataType::Float16 => Ok(input.to_vec()),
            DataType::Int8 => {
                let scale = input.iter().map(|x| x.abs()).fold(0.0, f32::max) / 127.0;
                Ok(input
                    .iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0))
                    .collect())
            }
            DataType::Int4 => {
                let scale = input.iter().map(|x| x.abs()).fold(0.0, f32::max) / 7.0;
                Ok(input
                    .iter()
                    .map(|&x| (x / scale).clamp(-7.0, 7.0))
                    .collect())
            }
        }
    }
    /// Fixed point inference.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn fixed_point_inference(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        let _ = kernel;
        let fixed_input: Vec<i32> = input.iter().map(|&x| (x * 65536.0) as i32).collect();

        let output: Vec<f32> = fixed_input.iter().map(|&x| x as f32 / 65536.0).collect();
        Ok(output)
    }
    /// Execute kernel.
    /// # Errors
    /// - Returns [`crate::KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(super) fn execute_kernel(
        &self,
        input: &[f32],
        kernel: &ExecutionKernel,
    ) -> KwaversResult<Vec<f32>> {
        let mut output = vec![0.0; kernel.io_spec.output_shape[0]];

        let Some(ref quantized_model) = self.model else {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "model".to_string(),
                    reason: "No quantized model loaded".to_string(),
                },
            ));
        };

        let layer_idx = kernel
            .id
            .strip_prefix("layer_")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "kernel.id".to_string(),
                    reason: format!("Invalid kernel id: {}", kernel.id),
                })
            })?;

        let layer = quantized_model
            .original_layers
            .get(layer_idx)
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "model.layers".to_string(),
                    reason: format!("Missing layer index {}", layer_idx),
                })
            })?;

        let weight_tensor = quantized_model
            .quantized_weights
            .get(layer_idx * 2)
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "model.quantized_weights".to_string(),
                    reason: format!("Missing weights for layer {}", layer_idx),
                })
            })?;
        let bias_tensor = quantized_model
            .quantized_weights
            .get(layer_idx * 2 + 1)
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "model.quantized_weights".to_string(),
                    reason: format!("Missing biases for layer {}", layer_idx),
                })
            })?;

        let input_len = (input.len()).min(layer.input_size);
        let output_len = (output.len()).min(layer.output_size);
        output.truncate(output_len);

        let weights = weight_tensor.dequantize();
        let biases = bias_tensor.dequantize();

        for out_idx in 0..(output.len()) {
            let mut sum = 0.0f32;
            for (j, &input_val) in input.iter().enumerate().take(input_len) {
                let weight_index = j * layer.output_size + out_idx;
                if weight_index < (weights.len()) {
                    sum += input_val * weights[weight_index];
                }
            }

            if out_idx < (biases.len()) {
                sum += biases[out_idx];
            }

            output[out_idx] = sum.tanh();
        }

        Ok(output)
    }
    /// Dequantize output.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn dequantize_output(&self, quantized_output: &[f32]) -> KwaversResult<Vec<f32>> {
        match self.hardware_caps.architecture {
            Architecture::ARM | Architecture::ARM64 => self.neon_dequantize(quantized_output),
            Architecture::RISCV => self.riscv_dequantize(quantized_output),
            _ => Ok(quantized_output.to_vec()),
        }
    }
    /// Neon dequantize.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn neon_dequantize(&self, input: &[f32]) -> KwaversResult<Vec<f32>> {
        Ok(input.to_vec())
    }
    /// Riscv dequantize.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn riscv_dequantize(&self, input: &[f32]) -> KwaversResult<Vec<f32>> {
        Ok(input.to_vec())
    }
}
