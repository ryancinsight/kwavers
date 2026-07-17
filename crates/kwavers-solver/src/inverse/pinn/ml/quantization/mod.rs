//! Model Quantization for PINN Optimization
//!
//! This module provides quantization strategies to reduce model size and improve inference
//! speed while maintaining physics-informed accuracy for real-time applications.

use std::collections::HashMap;

/// Quantization scheme configuration
#[derive(Debug, Clone)]
pub enum QuantizationScheme {
    None,
    Dynamic8Bit,
    Static8Bit {
        calibration_data: Vec<Vec<f32>>,
    },
    MixedPrecision {
        weight_bits: u8,
        activation_bits: u8,
    },
    Adaptive {
        accuracy_threshold: f32,
        max_bits: u8,
    },
}

/// Quantized model representation
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    pub original_layers: Vec<LayerInfo>,
    pub quantized_weights: Vec<QuantizedTensor>,
    pub quantization_params: QuantizationParams,
    pub metadata: QuantizationModelMetadata,
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
    F32(Vec<f32>),
    I8(Vec<i8>),
}

/// Quantized tensor with scale and zero point
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: QuantizedData,
    pub scale: f32,
    pub zero_point: i8,
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    pub fn len(&self) -> usize {
        match &self.data {
            QuantizedData::F32(v) => v.len(),
            QuantizedData::I8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        (self.len()) == 0
    }

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
    pub global_scale: f32,
    pub layer_scales: HashMap<String, f32>,
    pub scheme: QuantizationScheme,
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct QuantizationModelMetadata {
    pub original_accuracy: f32,
    pub quantized_accuracy: f32,
    pub compression_ratio: f32,
    pub inference_speedup: f32,
}

/// Quantization engine
#[derive(Debug)]
pub struct MlQuantizer {
    pub(super) scheme: QuantizationScheme,
    pub(super) calibration_samples: usize,
    pub(super) accuracy_tolerance: f32,
}

mod engine;
mod model;
#[cfg(test)]
mod tests;
