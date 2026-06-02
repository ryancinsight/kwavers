//! Quantization engine — public API and model-level operations.

use std::collections::HashMap;

use burn::tensor::backend::Backend;

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::inverse::pinn::ml::BurnPINN2DWave;

use crate::inverse::pinn::ml::quantization::{
    LayerInfo, MlQuantizer, QuantizationParams, QuantizationScheme, QuantizedModel, QuantizedTensor,
};

mod ops;

/// Quantization validation result (internal)
#[derive(Debug, Clone)]
pub(super) struct QuantizationValidationResult {
    pub original_accuracy: f32,
    pub quantized_accuracy: f32,
    pub accuracy_loss: f32,
    pub compression_ratio: f32,
}

impl MlQuantizer {
    /// Create a new quantizer with the given scheme.
    pub fn new(scheme: QuantizationScheme) -> Self {
        Self {
            scheme,
            calibration_samples: 1000,
            accuracy_tolerance: 0.05,
        }
    }

    /// Quantize a model, validating that accuracy loss stays within tolerance.
    /// # Errors
    /// - Returns [`KwaversError::System`] if accuracy loss exceeds tolerance.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
                kwavers_core::error::SystemError::InvalidConfiguration {
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

        let metadata = crate::inverse::pinn::ml::quantization::QuantizationModelMetadata {
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

    /// Compute global scale: maximum scale across all quantized weight tensors.
    pub(super) fn calculate_global_scale(&self, quantized_weights: &[QuantizedTensor]) -> f32 {
        quantized_weights
            .iter()
            .map(|t| t.scale)
            .fold(0.0f32, f32::max)
    }

    /// Compute per-layer scales from the first weight tensor for each layer.
    pub(super) fn calculate_layer_scales(
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

    /// Estimate inference speedup factor for the active scheme.
    pub(super) fn estimate_inference_speedup(&self) -> f32 {
        match &self.scheme {
            QuantizationScheme::None => 1.0,
            QuantizationScheme::Dynamic8Bit => 2.5,
            QuantizationScheme::Static8Bit { .. } => 3.0,
            QuantizationScheme::MixedPrecision { .. } => 2.0,
            QuantizationScheme::Adaptive { .. } => 2.8,
        }
    }
}
