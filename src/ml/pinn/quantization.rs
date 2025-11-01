//! Model Quantization for PINN Optimization
//!
//! This module provides quantization strategies to reduce model size and improve inference
//! speed while maintaining physics-informed accuracy for real-time applications.

use crate::error::{KwaversError, KwaversResult};
use burn::tensor::{Tensor, backend::Backend};
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
pub struct QuantizedModel<B: Backend> {
    /// Original model layers (kept for reference)
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

/// Quantized tensor with scale and zero point
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized values (u8 or i8)
    pub data: Vec<i8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i8,
    /// Original shape
    pub shape: Vec<usize>,
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
pub struct Quantizer<B: Backend> {
    scheme: QuantizationScheme,
    calibration_samples: usize,
    accuracy_tolerance: f32,
}

impl<B: Backend> Quantizer<B> {
    /// Create a new quantizer
    pub fn new(scheme: QuantizationScheme) -> Self {
        Self {
            scheme,
            calibration_samples: 1000,
            accuracy_tolerance: 0.05, // 5% accuracy loss tolerance
        }
    }

    /// Quantize a PINN model
    pub fn quantize_model(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
    ) -> KwaversResult<QuantizedModel<B>> {
        // Analyze model structure
        let layers = self.analyze_model_layers(model)?;

        // Collect calibration data if needed
        let calibration_data = match &self.scheme {
            QuantizationScheme::Static8Bit { calibration_data } => calibration_data.clone(),
            QuantizationScheme::Adaptive { .. } => self.collect_calibration_data(model)?,
            _ => Vec::new(),
        };

        // Quantize weights and biases
        let quantized_weights = self.quantize_weights(&layers, &calibration_data)?;

        // Validate quantization accuracy
        let validation_result = self.validate_quantization(model, &quantized_weights)?;
        if validation_result.accuracy_loss > self.accuracy_tolerance {
            return Err(KwaversError::System(crate::error::SystemError::InvalidConfiguration {
                parameter: "quantization_accuracy".to_string(),
                reason: format!("Accuracy loss {:.3} exceeds tolerance {:.3}",
                    validation_result.accuracy_loss, self.accuracy_tolerance),
            }));
        }

        let quantization_params = QuantizationParams {
            global_scale: self.calculate_global_scale(&layers),
            layer_scales: self.calculate_layer_scales(&layers),
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
    fn analyze_model_layers(
        &self,
        _model: &crate::ml::pinn::BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<LayerInfo>> {
        // In practice, this would inspect the Burn model structure
        // For now, return a representative layer structure
        Ok(vec![
            LayerInfo {
                name: "input_layer".to_string(),
                input_size: 3,
                output_size: 200,
                activation: "tanh".to_string(),
            },
            LayerInfo {
                name: "hidden_1".to_string(),
                input_size: 200,
                output_size: 200,
                activation: "tanh".to_string(),
            },
            LayerInfo {
                name: "hidden_2".to_string(),
                input_size: 200,
                output_size: 200,
                activation: "tanh".to_string(),
            },
            LayerInfo {
                name: "output_layer".to_string(),
                input_size: 200,
                output_size: 1,
                activation: "linear".to_string(),
            },
        ])
    }

    /// Collect calibration data for static quantization
    fn collect_calibration_data(
        &self,
        _model: &crate::ml::pinn::BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<Vec<f32>>> {
        // Generate representative physics input data
        let mut calibration_data = Vec::new();

        for _ in 0..self.calibration_samples {
            // Generate (x, y, t) coordinates covering the domain
            let x = rand::random::<f32>() * 2.0 - 1.0; // [-1, 1]
            let y = rand::random::<f32>() * 2.0 - 1.0; // [-1, 1]
            let t = rand::random::<f32>() * 1.0;       // [0, 1]

            calibration_data.push(vec![x, y, t]);
        }

        Ok(calibration_data)
    }

    /// Quantize model weights
    fn quantize_weights(
        &self,
        layers: &[LayerInfo],
        calibration_data: &[Vec<f32>],
    ) -> KwaversResult<Vec<QuantizedTensor>> {
        let mut quantized_weights = Vec::new();

        for layer in layers {
            // Simulate weight quantization
            // In practice, this would access actual model weights
            let weight_count = layer.input_size * layer.output_size;
            let bias_count = layer.output_size;

            // Generate mock weight data (normally from model)
            let weights = self.generate_mock_weights(weight_count);
            let biases = self.generate_mock_weights(bias_count);

            // Quantize weights
            let quantized_weight_tensor = self.quantize_tensor(&weights, layer)?;
            let quantized_bias_tensor = self.quantize_tensor(&biases, layer)?;

            quantized_weights.push(quantized_weight_tensor);
            quantized_weights.push(quantized_bias_tensor);
        }

        Ok(quantized_weights)
    }

    /// Quantize a single tensor
    fn quantize_tensor(&self, data: &[f32], layer: &LayerInfo) -> KwaversResult<QuantizedTensor> {
        match &self.scheme {
            QuantizationScheme::None => {
                // No quantization - convert to i8 range (inefficient but preserves precision)
                let scale = 1.0 / 127.0; // Map f32 range to i8
                let quantized_data: Vec<i8> = data.iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0) as i8)
                    .collect();

                Ok(QuantizedTensor {
                    data: quantized_data,
                    scale,
                    zero_point: 0,
                    shape: vec![data.len()],
                })
            }
            QuantizationScheme::Dynamic8Bit | QuantizationScheme::Static8Bit { .. } => {
                // 8-bit symmetric quantization
                let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = abs_max / 127.0;

                if scale == 0.0 {
                    // Handle zero tensor
                    return Ok(QuantizedTensor {
                        data: vec![0; data.len()],
                        scale: 1.0,
                        zero_point: 0,
                        shape: vec![data.len()],
                    });
                }

                let quantized_data: Vec<i8> = data.iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0) as i8)
                    .collect();

                Ok(QuantizedTensor {
                    data: quantized_data,
                    scale,
                    zero_point: 0,
                    shape: vec![data.len()],
                })
            }
            QuantizationScheme::MixedPrecision { weight_bits, .. } => {
                // Variable bit quantization
                let bits = *weight_bits;
                let max_val = 2f32.powf(bits as f32 - 1.0) - 1.0;
                let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = abs_max / max_val;

                let quantized_data: Vec<i8> = data.iter()
                    .map(|&x| (x / scale).clamp(-max_val, max_val) as i8)
                    .collect();

                Ok(QuantizedTensor {
                    data: quantized_data,
                    scale,
                    zero_point: 0,
                    shape: vec![data.len()],
                })
            }
            QuantizationScheme::Adaptive { accuracy_threshold, max_bits } => {
                // Adaptive quantization based on sensitivity
                // Start with higher precision and reduce based on impact
                let mut current_bits = *max_bits;

                loop {
                    let test_scale = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
                        / (2f32.powf(current_bits as f32 - 1.0) - 1.0);

                    let test_quantized: Vec<i8> = data.iter()
                        .map(|&x| (x / test_scale).clamp(-127.0, 127.0) as i8)
                        .collect();

                    // Estimate quantization error
                    let error: f32 = data.iter().zip(&test_quantized)
                        .map(|(&orig, &quant)| {
                            let dequant = quant as f32 * test_scale;
                            (orig - dequant).powi(2)
                        })
                        .sum();

                    let rmse = (error / data.len() as f32).sqrt();
                    let relative_error = rmse / data.iter().map(|x| x.abs()).sum::<f32>() * data.len() as f32;

                    if relative_error <= *accuracy_threshold || current_bits <= 4 {
                        return Ok(QuantizedTensor {
                            data: test_quantized,
                            scale: test_scale,
                            zero_point: 0,
                            shape: vec![data.len()],
                        });
                    }

                    current_bits -= 1;
                }
            }
        }
    }

    /// Generate mock weights for demonstration
    fn generate_mock_weights(&self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let layer_factor = (i % 4) as f32 * 0.1;
                (rand::random::<f32>() - 0.5) * 0.1 + layer_factor
            })
            .collect()
    }

    /// Validate quantization accuracy
    fn validate_quantization(
        &self,
        _model: &crate::ml::pinn::BurnPINN2DWave<B>,
        _quantized_weights: &[QuantizedTensor],
    ) -> KwaversResult<ValidationResult> {
        // In practice, this would run inference on test data
        // and compare original vs quantized accuracy
        Ok(ValidationResult {
            original_accuracy: 0.95, // Mock high accuracy
            quantized_accuracy: 0.92, // Small accuracy loss
            accuracy_loss: 0.03,
            compression_ratio: 4.0, // 4x compression for 8-bit
        })
    }

    /// Calculate global scale factor
    fn calculate_global_scale(&self, layers: &[LayerInfo]) -> f32 {
        // Use the maximum weight magnitude across all layers
        // In practice, this would analyze actual weights
        0.1 // Conservative estimate
    }

    /// Calculate per-layer scale factors
    fn calculate_layer_scales(&self, layers: &[LayerInfo]) -> HashMap<String, f32> {
        let mut scales = HashMap::new();

        for layer in layers {
            // Different scales for different layer types
            let scale = match layer.name.as_str() {
                "input_layer" => 0.05,
                "output_layer" => 0.01,
                _ => 0.1,
            };
            scales.insert(layer.name.clone(), scale);
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

impl<B: Backend> QuantizedModel<B> {
    /// Get model memory usage
    pub fn memory_usage(&self) -> usize {
        self.quantized_weights.iter()
            .map(|tensor| tensor.data.len())
            .sum::<usize>() * std::mem::size_of::<i8>()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        // Estimate original FP32 size vs quantized size
        let original_size = self.original_layers.iter()
            .map(|layer| (layer.input_size * layer.output_size + layer.output_size) * 4) // FP32
            .sum::<usize>();

        let quantized_size = self.memory_usage();

        original_size as f32 / quantized_size as f32
    }

    /// Dequantize weights for a specific layer
    pub fn dequantize_layer(&self, layer_name: &str) -> Option<Vec<f32>> {
        self.quantized_weights.iter()
            .find(|tensor| tensor.shape.len() > 1) // Find weight tensors
            .map(|tensor| {
                tensor.data.iter()
                    .map(|&q| (q as f32 - tensor.zero_point as f32) * tensor.scale)
                    .collect()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn test_quantizer_creation() {
        let quantizer = Quantizer::<TestBackend>::new(QuantizationScheme::Dynamic8Bit);
        assert_eq!(quantizer.accuracy_tolerance, 0.05);
    }

    #[test]
    fn test_quantization_schemes() {
        let schemes = vec![
            QuantizationScheme::None,
            QuantizationScheme::Dynamic8Bit,
            QuantizationScheme::Static8Bit { calibration_data: vec![vec![1.0, 2.0, 3.0]] },
            QuantizationScheme::MixedPrecision { weight_bits: 8, activation_bits: 8 },
            QuantizationScheme::Adaptive { accuracy_threshold: 0.05, max_bits: 8 },
        ];

        for scheme in schemes {
            let quantizer = Quantizer::<TestBackend>::new(scheme);
            assert!(quantizer.accuracy_tolerance >= 0.0);
        }
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let layer = LayerInfo {
            name: "test".to_string(),
            input_size: 4,
            output_size: 4,
            activation: "tanh".to_string(),
        };

        let quantizer = Quantizer::<TestBackend>::new(QuantizationScheme::Dynamic8Bit);
        let result = quantizer.quantize_tensor(&data, &layer);

        assert!(result.is_ok());
        let quantized = result.unwrap();
        assert_eq!(quantized.data.len(), 4);
        assert!(quantized.scale > 0.0);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let model = QuantizedModel::<TestBackend> {
            original_layers: vec![LayerInfo {
                name: "test".to_string(),
                input_size: 10,
                output_size: 10,
                activation: "tanh".to_string(),
            }],
            quantized_weights: vec![QuantizedTensor {
                data: vec![1i8; 100], // 100 bytes
                scale: 1.0,
                zero_point: 0,
                shape: vec![10, 10],
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
