use std::collections::HashMap;

use burn::tensor::{backend::Backend, Tensor};

use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::inverse::pinn::ml::BurnPINN2DWave;

use super::{
    LayerInfo, QuantizationParams, QuantizationScheme, QuantizedData, QuantizedModel,
    QuantizedTensor, Quantizer,
};

/// Quantization validation result (internal)
#[derive(Debug, Clone)]
struct ValidationResult {
    pub original_accuracy: f32,
    pub quantized_accuracy: f32,
    pub accuracy_loss: f32,
    pub compression_ratio: f32,
}

impl Quantizer {
    pub fn new(scheme: QuantizationScheme) -> Self {
        Self {
            scheme,
            calibration_samples: 1000,
            accuracy_tolerance: 0.05,
        }
    }

    pub fn quantize_model<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<QuantizedModel> {
        let layers = self.analyze_model_layers(model)?;

        let _calibration_data = match &self.scheme {
            QuantizationScheme::Static8Bit { calibration_data } => calibration_data.clone(),
            QuantizationScheme::Adaptive { .. } => self.collect_calibration_data(model)?,
            _ => Vec::new(),
        };

        let quantized_weights = self.quantize_weights(model)?;

        let validation_result = self.validate_quantization(model, &quantized_weights)?;
        if validation_result.accuracy_loss > self.accuracy_tolerance {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "quantization_accuracy".to_string(),
                    reason: format!(
                        "Accuracy loss {:.3} exceeds tolerance {:.3}",
                        validation_result.accuracy_loss, self.accuracy_tolerance
                    ),
                },
            ));
        }

        let quantization_params = QuantizationParams {
            global_scale: self.calculate_global_scale(&quantized_weights),
            layer_scales: self.calculate_layer_scales(&layers, &quantized_weights),
            scheme: self.scheme.clone(),
        };

        let metadata = super::ModelMetadata {
            original_accuracy: validation_result.original_accuracy,
            quantized_accuracy: validation_result.quantized_accuracy,
            compression_ratio: validation_result.compression_ratio,
            inference_speedup: self.estimate_inference_speedup(),
        };

        Ok(QuantizedModel {
            original_layers: layers,
            quantized_weights,
            quantization_params,
            metadata,
        })
    }

    fn analyze_model_layers<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<LayerInfo>> {
        let mut layers = Vec::new();

        let input_shape = model.input_layer.weight.val().shape();
        layers.push(LayerInfo {
            name: "input_layer".to_string(),
            input_size: input_shape.dims[1],
            output_size: input_shape.dims[0],
            activation: "tanh".to_string(),
        });

        for (i, layer) in model.hidden_layers.iter().enumerate() {
            let shape = layer.weight.val().shape();
            layers.push(LayerInfo {
                name: format!("hidden_{}", i),
                input_size: shape.dims[1],
                output_size: shape.dims[0],
                activation: "tanh".to_string(),
            });
        }

        let output_shape = model.output_layer.weight.val().shape();
        layers.push(LayerInfo {
            name: "output_layer".to_string(),
            input_size: output_shape.dims[1],
            output_size: output_shape.dims[0],
            activation: "linear".to_string(),
        });

        Ok(layers)
    }

    fn collect_calibration_data<B: Backend>(
        &self,
        _model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<Vec<f32>>> {
        let mut calibration_data = Vec::new();

        for _ in 0..self.calibration_samples {
            let x = rand::random::<f32>() * 2.0 - 1.0;
            let y = rand::random::<f32>() * 2.0 - 1.0;
            let t = rand::random::<f32>() * 1.0;
            calibration_data.push(vec![x, y, t]);
        }

        Ok(calibration_data)
    }

    fn quantize_weights<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<QuantizedTensor>> {
        let mut quantized_weights = Vec::new();

        quantized_weights.push(self.quantize_burn_tensor(&model.input_layer.weight.val())?);
        if let Some(bias) = &model.input_layer.bias {
            quantized_weights.push(self.quantize_burn_tensor(&bias.val())?);
        }

        for layer in &model.hidden_layers {
            quantized_weights.push(self.quantize_burn_tensor(&layer.weight.val())?);
            if let Some(bias) = &layer.bias {
                quantized_weights.push(self.quantize_burn_tensor(&bias.val())?);
            }
        }

        quantized_weights.push(self.quantize_burn_tensor(&model.output_layer.weight.val())?);
        if let Some(bias) = &model.output_layer.bias {
            quantized_weights.push(self.quantize_burn_tensor(&bias.val())?);
        }

        Ok(quantized_weights)
    }

    fn quantize_burn_tensor<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> KwaversResult<QuantizedTensor> {
        let data = tensor.to_data();
        let floats = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "tensor_data".to_string(),
                reason: format!("Expected f32 tensor data: {:?}", e),
            })
        })?;

        let shape = tensor.shape().dims.to_vec();
        self.quantize_tensor(floats, &shape)
    }

    pub(super) fn quantize_tensor(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> KwaversResult<QuantizedTensor> {
        match &self.scheme {
            QuantizationScheme::None => Ok(QuantizedTensor {
                data: QuantizedData::F32(data.to_vec()),
                scale: 1.0,
                zero_point: 0,
                shape: shape.to_vec(),
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
                    shape: shape.to_vec(),
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
                    shape: shape.to_vec(),
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
                    let rmse = (error / data.len() as f32).sqrt();
                    let sum_abs = data.iter().map(|x| x.abs()).sum::<f32>();
                    let relative_error = if sum_abs == 0.0 {
                        0.0
                    } else {
                        rmse / (sum_abs / data.len() as f32)
                    };
                    if relative_error <= *accuracy_threshold || current_bits <= 4 {
                        return Ok(QuantizedTensor {
                            data: QuantizedData::I8(test_quantized),
                            scale: test_scale,
                            zero_point: 0,
                            shape: shape.to_vec(),
                        });
                    }
                    current_bits -= 1;
                }
            }
        }
    }

    fn validate_quantization<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
        quantized_weights: &[QuantizedTensor],
    ) -> KwaversResult<ValidationResult> {
        let original_params = model.parameters();

        if original_params.len() != quantized_weights.len() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "quantized_weights".to_string(),
                    reason: format!(
                        "Parameter count mismatch: original={}, quantized={}",
                        original_params.len(),
                        quantized_weights.len()
                    ),
                },
            ));
        }

        let mut total_mse = 0.0;
        let mut total_elements = 0;
        let mut original_bytes = 0;
        let mut quantized_bytes = 0;

        for (orig, quant) in original_params.iter().zip(quantized_weights) {
            let orig_data = orig.to_data();
            let orig_floats = orig_data.as_slice::<f32>().map_err(|e| {
                KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "original_params".to_string(),
                    reason: format!("Failed to get f32 slice: {:?}", e),
                })
            })?;

            let dequant_floats = quant.dequantize();
            original_bytes += orig_floats.len() * 4;

            match &quant.data {
                QuantizedData::F32(v) => quantized_bytes += v.len() * 4,
                QuantizedData::I8(v) => quantized_bytes += v.len(),
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

        Ok(ValidationResult {
            original_accuracy: 1.0,
            quantized_accuracy: (1.0 - rmse).max(0.0),
            accuracy_loss,
            compression_ratio,
        })
    }

    fn calculate_global_scale(&self, quantized_weights: &[QuantizedTensor]) -> f32 {
        quantized_weights
            .iter()
            .map(|t| t.scale)
            .fold(0.0f32, f32::max)
    }

    fn calculate_layer_scales(
        &self,
        layers: &[LayerInfo],
        quantized_weights: &[QuantizedTensor],
    ) -> HashMap<String, f32> {
        let mut scales = HashMap::new();
        let mut weight_idx = 0;

        for layer in layers {
            let weight_scale = quantized_weights[weight_idx].scale;
            scales.insert(layer.name.clone(), weight_scale);
            weight_idx += 2;
        }

        scales
    }

    fn estimate_inference_speedup(&self) -> f32 {
        match &self.scheme {
            QuantizationScheme::None => 1.0,
            QuantizationScheme::Dynamic8Bit => 2.5,
            QuantizationScheme::Static8Bit { .. } => 3.0,
            QuantizationScheme::MixedPrecision { .. } => 2.0,
            QuantizationScheme::Adaptive { .. } => 2.8,
        }
    }
}
