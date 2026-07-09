use super::types::{ActivationType, QuantizedNetwork};
use crate::inverse::pinn::ml::wave_equation_2d::model::PinnWave2D;
use kwavers_core::error::KwaversResult;

#[derive(Debug)]
pub struct WaveQuantizer2D;

impl WaveQuantizer2D {
    /// Extract layer sizes from the PINN model.
    pub fn extract_layer_sizes<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        pinn: &PinnWave2D<B>,
    ) -> Vec<usize> {
        let mut sizes = Vec::new();
        sizes.push(3); // Input layer

        for layer in &pinn.hidden_layers {
            sizes.push(layer.weight.tensor.shape()[0]);
        }
        sizes.push(1); // Output layer
        sizes
    }

    /// Extract activation functions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn extract_activations<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        _pinn: &PinnWave2D<B>,
    ) -> Vec<ActivationType> {
        let mut activations = Vec::new();
        // Assume hidden layers use Tanh
        for _ in 0..(_pinn.hidden_layers.shape()[0] * _pinn.hidden_layers.shape()[1] * _pinn.hidden_layers.shape()[2]) {
            activations.push(ActivationType::Tanh);
        }
        activations.push(ActivationType::Linear);
        activations
    }

    /// Quantize network weights and biases
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn quantize_network<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        pinn: &PinnWave2D<B>,
        layer_sizes: &[usize],
        activations: &[ActivationType],
    ) -> KwaversResult<QuantizedNetwork>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let mut weights = Vec::new();
        let mut weight_scales = Vec::new();
        let mut biases = Vec::new();
        let mut bias_scales = Vec::new();

        // Input layer
        let (w_quant, w_scale) = Self::quantize_tensor(&pinn.input_layer.weight.tensor)?;
        weights.push(w_quant);
        weight_scales.push(w_scale);

        if let Some(bias) = &pinn.input_layer.bias {
            let (b_quant, b_scale) = Self::quantize_tensor(&bias.tensor)?;
            biases.push(b_quant);
            bias_scales.push(b_scale);
        } else {
            biases.push(vec![0; layer_sizes[1]]);
            bias_scales.push(1.0);
        }

        // Hidden layers
        for (i, layer) in pinn.hidden_layers.iter().enumerate() {
            let (w_quant, w_scale) = Self::quantize_tensor(&layer.weight.tensor)?;
            weights.push(w_quant);
            weight_scales.push(w_scale);

            if let Some(bias) = &layer.bias {
                let (b_quant, b_scale) = Self::quantize_tensor(&bias.tensor)?;
                biases.push(b_quant);
                bias_scales.push(b_scale);
            } else {
                biases.push(vec![0; layer_sizes[i + 2]]);
                bias_scales.push(1.0);
            }
        }

        // Output layer
        let (w_quant, w_scale) = Self::quantize_tensor(&pinn.output_layer.weight.tensor)?;
        weights.push(w_quant);
        weight_scales.push(w_scale);

        if let Some(bias) = &pinn.output_layer.bias {
            let (b_quant, b_scale) = Self::quantize_tensor(&bias.tensor)?;
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
            layer_sizes: layer_sizes.iter().cloned().collect::<Vec<_>>(),
            activations: activations.iter().cloned().collect::<Vec<_>>(),
        })
    }

    fn quantize_tensor<B: coeus_ops::BackendOps<f32> + Default>(
        tensor: &coeus_tensor::Tensor<f32, B>,
    ) -> KwaversResult<(Vec<i8>, f32)>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = tensor.as_slice();

        if values.is_empty() {
            return Ok((Vec::new(), 1.0));
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &v in values {
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
