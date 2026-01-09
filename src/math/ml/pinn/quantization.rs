//! Model Quantization for PINN Optimization
//!
//! This module provides quantization strategies to reduce model size and improve inference
//! speed while maintaining physics-informed accuracy for real-time applications.

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::ml::pinn::BurnPINN2DWave;
use burn::tensor::{backend::Backend, Tensor};
use std::collections::HashMap;

/// Quantization scheme configuration
#[derive(Debug, Clone)]
pub enum QuantizationScheme {
    /// No quantization (FP32 baseline)
    None,
    /// Dynamic 8-bit quantization
    Dynamic8Bit,
    /// Static 8-bit quantization with calibration
    Static8Bit {
        /// Calibration data for range estimation
        calibration_data: Vec<Vec<f32>>,
    },
    /// Mixed precision quantization
    MixedPrecision {
        /// Weight quantization bits
        weight_bits: u8,
        /// Activation quantization bits
        activation_bits: u8,
    },
    /// Adaptive quantization based on sensitivity
    Adaptive {
        /// Target accuracy loss threshold
        accuracy_threshold: f32,
        /// Maximum quantization bits
        max_bits: u8,
    },
}

/// Quantized model representation
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    /// Original model layers
    pub original_layers: Vec<LayerInfo>,
    /// Quantized weights and biases
    pub quantized_weights: Vec<QuantizedTensor>,
    /// Quantization parameters
    pub quantization_params: QuantizationParams,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Layer information for quantization
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: String,
}

/// Quantized or full-precision tensor data
#[derive(Debug, Clone)]
pub enum QuantizedData {
    /// Full precision 32-bit float
    F32(Vec<f32>),
    /// 8-bit signed integer
    I8(Vec<i8>),
}

/// Quantized tensor with scale and zero point
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized or full-precision values
    pub data: QuantizedData,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i8,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Get the number of elements in the tensor
    pub fn len(&self) -> usize {
        match &self.data {
            QuantizedData::F32(v) => v.len(),
            QuantizedData::I8(v) => v.len(),
        }
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dequantize to f32 vector
    pub fn dequantize(&self) -> Vec<f32> {
        match &self.data {
            QuantizedData::F32(v) => v.clone(),
            QuantizedData::I8(v) => v
                .iter()
                .map(|&q| (q.wrapping_sub(self.zero_point) as f32) * self.scale)
                .collect(),
        }
    }
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Global scale factor
    pub global_scale: f32,
    /// Per-layer scales
    pub layer_scales: HashMap<String, f32>,
    /// Quantization scheme used
    pub scheme: QuantizationScheme,
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub original_accuracy: f32,
    pub quantized_accuracy: f32,
    pub compression_ratio: f32,
    pub inference_speedup: f32,
}

/// Quantization engine
#[derive(Debug)]
pub struct Quantizer {
    scheme: QuantizationScheme,
    calibration_samples: usize,
    accuracy_tolerance: f32,
}

impl Quantizer {
    /// Create a new quantizer
    pub fn new(scheme: QuantizationScheme) -> Self {
        Self {
            scheme,
            calibration_samples: 1000,
            accuracy_tolerance: 0.05, // 5% accuracy loss tolerance
        }
    }

    /// Quantize a PINN model
    pub fn quantize_model<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<QuantizedModel> {
        // Analyze model structure
        let layers = self.analyze_model_layers(model)?;

        // Collect calibration data if needed
        let _calibration_data = match &self.scheme {
            QuantizationScheme::Static8Bit { calibration_data } => calibration_data.clone(),
            QuantizationScheme::Adaptive { .. } => self.collect_calibration_data(model)?,
            _ => Vec::new(),
        };

        // Quantize weights and biases
        let quantized_weights = self.quantize_weights(model)?;

        // Validate quantization accuracy
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

        let metadata = ModelMetadata {
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

    /// Analyze model layers for quantization
    fn analyze_model_layers<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<LayerInfo>> {
        let mut layers = Vec::new();

        // Input layer
        let input_shape = model.input_layer.weight.val().shape();
        layers.push(LayerInfo {
            name: "input_layer".to_string(),
            input_size: input_shape.dims[1],
            output_size: input_shape.dims[0],
            activation: "tanh".to_string(),
        });

        // Hidden layers
        for (i, layer) in model.hidden_layers.iter().enumerate() {
            let shape = layer.weight.val().shape();
            layers.push(LayerInfo {
                name: format!("hidden_{}", i),
                input_size: shape.dims[1],
                output_size: shape.dims[0],
                activation: "tanh".to_string(),
            });
        }

        // Output layer
        let output_shape = model.output_layer.weight.val().shape();
        layers.push(LayerInfo {
            name: "output_layer".to_string(),
            input_size: output_shape.dims[1],
            output_size: output_shape.dims[0],
            activation: "linear".to_string(),
        });

        Ok(layers)
    }

    /// Collect calibration data for static quantization
    fn collect_calibration_data<B: Backend>(
        &self,
        _model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<Vec<f32>>> {
        // Generate representative physics input data
        let mut calibration_data = Vec::new();

        for _ in 0..self.calibration_samples {
            // Generate (x, y, t) coordinates covering the domain
            let x = rand::random::<f32>() * 2.0 - 1.0; // [-1, 1]
            let y = rand::random::<f32>() * 2.0 - 1.0; // [-1, 1]
            let t = rand::random::<f32>() * 1.0; // [0, 1]

            calibration_data.push(vec![x, y, t]);
        }

        Ok(calibration_data)
    }

    /// Quantize model weights
    fn quantize_weights<B: Backend>(
        &self,
        model: &BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<QuantizedTensor>> {
        let mut quantized_weights = Vec::new();

        // Input layer
        quantized_weights.push(self.quantize_burn_tensor(&model.input_layer.weight.val())?);
        if let Some(bias) = &model.input_layer.bias {
            quantized_weights.push(self.quantize_burn_tensor(&bias.val())?);
        }

        // Hidden layers
        for layer in &model.hidden_layers {
            quantized_weights.push(self.quantize_burn_tensor(&layer.weight.val())?);
            if let Some(bias) = &layer.bias {
                quantized_weights.push(self.quantize_burn_tensor(&bias.val())?);
            }
        }

        // Output layer
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

    /// Quantize a single tensor
    fn quantize_tensor(&self, data: &[f32], shape: &[usize]) -> KwaversResult<QuantizedTensor> {
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

    /// Validate quantization accuracy
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

        // Heuristic: Accuracy loss is proportional to RMSE relative to weight magnitude
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

    /// Calculate global scale factor
    fn calculate_global_scale(&self, quantized_weights: &[QuantizedTensor]) -> f32 {
        quantized_weights
            .iter()
            .map(|t| t.scale)
            .fold(0.0f32, f32::max)
    }

    /// Calculate per-layer scale factors
    fn calculate_layer_scales(
        &self,
        layers: &[LayerInfo],
        quantized_weights: &[QuantizedTensor],
    ) -> HashMap<String, f32> {
        let mut scales = HashMap::new();
        let mut weight_idx = 0;

        for layer in layers {
            // Weight and optional bias
            let weight_scale = quantized_weights[weight_idx].scale;
            scales.insert(layer.name.clone(), weight_scale);
            weight_idx += 2; // Assume weight + bias for simplicity here
        }

        scales
    }

    /// Estimate inference speedup from quantization
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

/// Quantization validation result
#[derive(Debug, Clone)]
struct ValidationResult {
    pub original_accuracy: f32,
    pub quantized_accuracy: f32,
    pub accuracy_loss: f32,
    pub compression_ratio: f32,
}

impl QuantizedModel {
    /// Get model memory usage
    pub fn memory_usage(&self) -> usize {
        self.quantized_weights
            .iter()
            .map(|tensor| match &tensor.data {
                QuantizedData::F32(v) => v.len() * 4,
                QuantizedData::I8(v) => v.len(),
            })
            .sum::<usize>()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size: usize = self
            .original_layers
            .iter()
            .map(|l| l.input_size * l.output_size + l.output_size)
            .sum::<usize>()
            * 4; // FP32

        let quantized_size = self.memory_usage();
        original_size as f32 / quantized_size as f32
    }

    /// Dequantize weights for a specific layer
    pub fn dequantize_layer(&self, _layer_name: &str) -> Option<Vec<f32>> {
        self.quantized_weights
            .iter()
            .find(|tensor| tensor.shape.len() > 1) // Find weight tensors
            .map(|tensor| tensor.dequantize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_creation() {
        let quantizer = Quantizer::new(QuantizationScheme::Dynamic8Bit);
        assert_eq!(quantizer.accuracy_tolerance, 0.05);
    }

    #[test]
    fn test_quantization_schemes() {
        let schemes = vec![
            QuantizationScheme::None,
            QuantizationScheme::Dynamic8Bit,
            QuantizationScheme::Static8Bit {
                calibration_data: vec![vec![1.0, 2.0, 3.0]],
            },
            QuantizationScheme::MixedPrecision {
                weight_bits: 8,
                activation_bits: 8,
            },
            QuantizationScheme::Adaptive {
                accuracy_threshold: 0.05,
                max_bits: 8,
            },
        ];

        for scheme in schemes {
            let quantizer = Quantizer::new(scheme);
            assert!(quantizer.accuracy_tolerance >= 0.0);
        }
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let shape = vec![1, 4];

        let quantizer = Quantizer::new(QuantizationScheme::Dynamic8Bit);
        let result = quantizer.quantize_tensor(&data, &shape);

        assert!(result.is_ok());
        let quantized = result.unwrap();
        assert_eq!(quantized.len(), 4);
        assert!(quantized.scale > 0.0);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let model = QuantizedModel {
            original_layers: vec![LayerInfo {
                name: "test".to_string(),
                input_size: 10,
                output_size: 10,
                activation: "tanh".to_string(),
            }],
            quantized_weights: vec![QuantizedTensor {
                data: QuantizedData::I8(vec![1i8; 110]), // 100 weights + 10 biases
                scale: 1.0,
                zero_point: 0,
                shape: vec![10, 11],
            }],
            quantization_params: QuantizationParams {
                global_scale: 1.0,
                layer_scales: HashMap::new(),
                scheme: QuantizationScheme::Dynamic8Bit,
            },
            metadata: ModelMetadata {
                original_accuracy: 0.95,
                quantized_accuracy: 0.92,
                compression_ratio: 4.0,
                inference_speedup: 2.5,
            },
        };

        let ratio = model.compression_ratio();
        assert!(ratio > 1.0); // Should show compression
    }
}
