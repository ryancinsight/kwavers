//! Per-layer and per-tensor quantization operations for [`MlQuantizer`].

use crate::inverse::pinn::ml::PinnWave2D;
use kwavers_core::error::{KwaversError, KwaversResult};

use crate::inverse::pinn::ml::quantization::{
    LayerInfo, MlQuantizer, QuantizationScheme, QuantizedData, QuantizedTensor,
};

use super::QuantizationValidationResult;

impl MlQuantizer {
    /// Enumerate model layer shapes and activation types.
    pub(super) fn analyze_model_layers<
        B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    >(
        &self,
        model: &PinnWave2D<B>,
    ) -> KwaversResult<Vec<LayerInfo>> {
        let mut layers = Vec::new();

        // coeus Linear weight layout is `[out_features, in_features]`.
        let input_shape = model.input_layer.weight.tensor.shape();
        layers.push(LayerInfo {
            name: "input_layer".to_string(),
            input_size: input_shape[1],
            output_size: input_shape[0],
            activation: "tanh".to_string(),
        });

        for (i, layer) in model.hidden_layers.iter().enumerate() {
            let shape = layer.weight.tensor.shape();
            layers.push(LayerInfo {
                name: format!("hidden_{}", i),
                input_size: shape[1],
                output_size: shape[0],
                activation: "tanh".to_string(),
            });
        }

        let output_shape = model.output_layer.weight.tensor.shape();
        layers.push(LayerInfo {
            name: "output_layer".to_string(),
            input_size: output_shape[1],
            output_size: output_shape[0],
            activation: "linear".to_string(),
        });

        Ok(layers)
    }

    /// Collect calibration inputs by uniform random sampling.
    pub(super) fn collect_calibration_data<
        B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    >(
        &self,
        _model: &PinnWave2D<B>,
    ) -> KwaversResult<Vec<Vec<f32>>> {
        let mut calibration_data = Vec::new();

        for _ in 0..self.calibration_samples {
            let x = rand::random::<f32>() * 2.0 - 1.0;
            let y = rand::random::<f32>() * 2.0 - 1.0;
            let t = rand::random::<f32>();
            calibration_data.push(vec![x, y, t]);
        }

        Ok(calibration_data)
    }

    /// Quantize all weight and bias tensors in model order.
    pub(super) fn quantize_weights<
        B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    >(
        &self,
        model: &PinnWave2D<B>,
    ) -> KwaversResult<Vec<QuantizedTensor>>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let mut quantized_weights = Vec::new();

        quantized_weights.push(self.quantize_coeus_tensor(&model.input_layer.weight.tensor)?);
        if let Some(bias) = &model.input_layer.bias {
            quantized_weights.push(self.quantize_coeus_tensor(&bias.tensor)?);
        }

        for layer in &model.hidden_layers {
            quantized_weights.push(self.quantize_coeus_tensor(&layer.weight.tensor)?);
            if let Some(bias) = &layer.bias {
                quantized_weights.push(self.quantize_coeus_tensor(&bias.tensor)?);
            }
        }

        quantized_weights.push(self.quantize_coeus_tensor(&model.output_layer.weight.tensor)?);
        if let Some(bias) = &model.output_layer.bias {
            quantized_weights.push(self.quantize_coeus_tensor(&bias.tensor)?);
        }

        Ok(quantized_weights)
    }

    /// Convert a coeus tensor to a `QuantizedTensor` by extracting f32 data.
    pub(super) fn quantize_coeus_tensor<
        B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    >(
        &self,
        tensor: &coeus_tensor::Tensor<f32, B>,
    ) -> KwaversResult<QuantizedTensor>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let floats = tensor.as_slice();
        let shape = tensor.shape().to_vec();
        self.quantize_tensor(floats, &shape)
    }

    /// Quantize a flat f32 slice according to the active scheme.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn quantize_tensor(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> KwaversResult<QuantizedTensor> {
        match &self.scheme {
            QuantizationScheme::None => Ok(QuantizedTensor {
                data: QuantizedData::F32(data.iter().cloned().collect::<Vec<_>>()),
                scale: 1.0,
                zero_point: 0,
                shape: shape.iter().cloned().collect::<Vec<_>>(),
            }),
            QuantizationScheme::Dynamic8Bit | QuantizationScheme::Static8Bit { .. } => {
                let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
                let quantized_data: Vec<i8> = data
                    .iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0) as i8)
                    .collect();
                Ok(QuantizedTensor {
                    data: QuantizedData::I8(quantized_data),
                    scale,
                    zero_point: 0,
                    shape: shape.iter().cloned().collect::<Vec<_>>(),
                })
            }
            QuantizationScheme::MixedPrecision { weight_bits, .. } => {
                let bits = *weight_bits;
                let max_val = 2f32.powf(bits as f32 - 1.0) - 1.0;
                let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = if abs_max == 0.0 {
                    1.0
                } else {
                    abs_max / max_val
                };
                let quantized_data: Vec<i8> = data
                    .iter()
                    .map(|&x| (x / scale).clamp(-max_val, max_val) as i8)
                    .collect();
                Ok(QuantizedTensor {
                    data: QuantizedData::I8(quantized_data),
                    scale,
                    zero_point: 0,
                    shape: shape.iter().cloned().collect::<Vec<_>>(),
                })
            }
            QuantizationScheme::Adaptive {
                accuracy_threshold,
                max_bits,
            } => {
                let mut current_bits = *max_bits;
                loop {
                    let max_val = 2f32.powf(current_bits as f32 - 1.0) - 1.0;
                    let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let test_scale = if abs_max == 0.0 {
                        1.0
                    } else {
                        abs_max / max_val
                    };
                    let test_quantized: Vec<i8> = data
                        .iter()
                        .map(|&x| (x / test_scale).clamp(-max_val, max_val) as i8)
                        .collect();
                    let error: f32 = data
                        .iter()
                        .zip(&test_quantized)
                        .map(|(&orig, &quant)| {
                            let dequant = quant as f32 * test_scale;
                            (orig - dequant).powi(2)
                        })
                        .sum();
                    let rmse = (error / (data.shape()[0] * data.shape()[1] * data.shape()[2]) as f32).sqrt();
                    let sum_abs = data.iter().map(|x| x.abs()).sum::<f32>();
                    let relative_error = if sum_abs == 0.0 {
                        0.0
                    } else {
                        rmse / (sum_abs / (data.shape()[0] * data.shape()[1] * data.shape()[2]) as f32)
                    };
                    if relative_error <= *accuracy_threshold || current_bits <= 4 {
                        return Ok(QuantizedTensor {
                            data: QuantizedData::I8(test_quantized),
                            scale: test_scale,
                            zero_point: 0,
                            shape: shape.iter().cloned().collect::<Vec<_>>(),
                        });
                    }
                    current_bits -= 1;
                }
            }
        }
    }

    /// Validate quantization accuracy by comparing original and dequantized tensors.
    /// # Errors
    /// - Returns [`KwaversError::System`] on parameter count mismatch.
    ///
    pub(super) fn validate_quantization<
        B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
    >(
        &self,
        model: &PinnWave2D<B>,
        quantized_weights: &[QuantizedTensor],
    ) -> KwaversResult<QuantizationValidationResult>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let original_params = model.parameters();

        if (original_params.shape()[0] * original_params.shape()[1] * original_params.shape()[2]) != (quantized_weights.shape()[0] * quantized_weights.shape()[1] * quantized_weights.shape()[2]) {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidConfiguration {
                    parameter: "quantized_weights".to_string(),
                    reason: format!(
                        "Parameter count mismatch: original={}, quantized={}",
                        (original_params.shape()[0] * original_params.shape()[1] * original_params.shape()[2]),
                        (quantized_weights.shape()[0] * quantized_weights.shape()[1] * quantized_weights.shape()[2])
                    ),
                },
            ));
        }

        let mut total_mse = 0.0f32;
        let mut total_elements = 0usize;
        let mut original_bytes = 0usize;
        let mut quantized_bytes = 0usize;

        for (orig, quant) in original_params.iter().zip(quantized_weights) {
            let orig_floats = orig.tensor.as_slice();

            let dequant_floats = quant.dequantize();
            original_bytes += (orig_floats.shape()[0] * orig_floats.shape()[1] * orig_floats.shape()[2]) * 4;

            match &quant.data {
                QuantizedData::F32(v) => quantized_bytes += (v.shape()[0] * v.shape()[1] * v.shape()[2]) * 4,
                QuantizedData::I8(v) => quantized_bytes += (v.shape()[0] * v.shape()[1] * v.shape()[2]),
            }

            for (o, q) in orig_floats.iter().zip(&dequant_floats) {
                total_mse += (o - q).powi(2);
                total_elements += 1;
            }
        }

        let rmse = if total_elements > 0 {
            (total_mse / total_elements as f32).sqrt()
        } else {
            0.0
        };

        let accuracy_loss = (rmse * 100.0).clamp(0.0, 100.0);
        let compression_ratio = if quantized_bytes > 0 {
            original_bytes as f32 / quantized_bytes as f32
        } else {
            1.0
        };

        Ok(QuantizationValidationResult {
            original_accuracy: 1.0,
            quantized_accuracy: (1.0 - rmse).max(0.0),
            accuracy_loss,
            compression_ratio,
        })
    }
}
