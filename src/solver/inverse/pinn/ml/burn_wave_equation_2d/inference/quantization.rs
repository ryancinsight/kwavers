use super::types::{ActivationType, QuantizedNetwork};
use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::model::BurnPINN2DWave;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Debug)]
pub struct Quantizer;

impl Quantizer {
    /// Extract layer sizes from Burn PINN model
    pub fn extract_layer_sizes<B: Backend>(pinn: &BurnPINN2DWave<B>) -> Vec<usize> {
        let mut sizes = Vec::new();
        sizes.push(3); // Input layer

        for layer in &pinn.hidden_layers {
            sizes.push(layer.weight.dims()[1]);
        }
        sizes.push(1); // Output layer
        sizes
    }

    /// Extract activation functions
    pub fn extract_activations<B: Backend>(_pinn: &BurnPINN2DWave<B>) -> Vec<ActivationType> {
        let mut activations = Vec::new();
        // Assume hidden layers use Tanh
        for _ in 0.._pinn.hidden_layers.len() {
            activations.push(ActivationType::Tanh);
        }
        activations.push(ActivationType::Linear);
        activations
    }

    /// Quantize network weights and biases
    pub fn quantize_network<B: Backend>(
        pinn: &BurnPINN2DWave<B>,
        layer_sizes: &[usize],
        activations: &[ActivationType],
    ) -> KwaversResult<QuantizedNetwork> {
        let mut weights = Vec::new();
        let mut weight_scales = Vec::new();
        let mut biases = Vec::new();
        let mut bias_scales = Vec::new();

        // Input layer
        let (w_quant, w_scale) = Self::quantize_tensor(&pinn.input_layer.weight.val())?;
        weights.push(w_quant);
        weight_scales.push(w_scale);

        if let Some(bias) = &pinn.input_layer.bias {
            let (b_quant, b_scale) = Self::quantize_tensor(&bias.val())?;
            biases.push(b_quant);
            bias_scales.push(b_scale);
        } else {
            biases.push(vec![0; layer_sizes[1]]);
            bias_scales.push(1.0);
        }

        // Hidden layers
        for (i, layer) in pinn.hidden_layers.iter().enumerate() {
            let (w_quant, w_scale) = Self::quantize_tensor(&layer.weight.val())?;
            weights.push(w_quant);
            weight_scales.push(w_scale);

            if let Some(bias) = &layer.bias {
                let (b_quant, b_scale) = Self::quantize_tensor(&bias.val())?;
                biases.push(b_quant);
                bias_scales.push(b_scale);
            } else {
                biases.push(vec![0; layer_sizes[i + 2]]);
                bias_scales.push(1.0);
            }
        }

        // Output layer
        let (w_quant, w_scale) = Self::quantize_tensor(&pinn.output_layer.weight.val())?;
        weights.push(w_quant);
        weight_scales.push(w_scale);

        if let Some(bias) = &pinn.output_layer.bias {
            let (b_quant, b_scale) = Self::quantize_tensor(&bias.val())?;
            biases.push(b_quant);
            bias_scales.push(b_scale);
        } else {
            biases.push(vec![0; layer_sizes.last().cloned().unwrap_or(1)]);
            bias_scales.push(1.0);
        }

        Ok(QuantizedNetwork {
            weights,
            weight_scales,
            biases,
            bias_scales,
            layer_sizes: layer_sizes.to_vec(),
            activations: activations.to_vec(),
        })
    }

    fn quantize_tensor<B: Backend, const D: usize>(
        tensor: &Tensor<B, D>,
    ) -> KwaversResult<(Vec<i8>, f32)> {
        let data = tensor.clone().into_data();
        let values: Vec<f32> = data.to_vec().unwrap_or_default();

        if values.is_empty() {
            return Ok((Vec::new(), 1.0));
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &v in &values {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };

        let quantized: Vec<i8> = values
            .iter()
            .map(|&v| {
                if scale > 0.0 {
                    let normalized = (v - min_val) / scale;
                    (normalized.clamp(0.0, 255.0) - 128.0) as i8
                } else {
                    0
                }
            })
            .collect();

        Ok((quantized, scale))
    }
}
