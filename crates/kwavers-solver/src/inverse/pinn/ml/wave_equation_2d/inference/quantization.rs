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
        let mut sizes = Vec::with_capacity(pinn.hidden_layers.len() + 3);
        sizes.push(3); // Coordinate input: (x, y, t)
        sizes.push(pinn.input_layer.weight.tensor.shape()[0]);

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
        let mut activations = Vec::with_capacity(_pinn.hidden_layers.len() + 2);
        // `PinnWave2D::forward` applies the first hidden projection directly,
        // then applies tanh after each subsequent hidden projection.
        activations.push(ActivationType::Linear);
        for _ in &_pinn.hidden_layers {
            activations.push(ActivationType::Tanh);
        }
        activations.push(ActivationType::Linear);
        activations
    }

    /// Quantize network weights and biases
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
            layer_sizes: layer_sizes.to_vec(),
            activations: activations.to_vec(),
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

        let max_magnitude = values
            .iter()
            .fold(0.0_f32, |maximum, &value| maximum.max(value.abs()));
        let scale = if max_magnitude > 0.0 {
            max_magnitude / 127.0
        } else {
            0.0
        };

        let quantized: Vec<i8> = values
            .iter()
            .map(|&v| {
                if scale > 0.0 {
                    (v / scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                }
            })
            .collect();

        Ok((quantized, scale))
    }
}

#[cfg(test)]
mod tests {
    use super::WaveQuantizer2D;
    use crate::inverse::pinn::ml::wave_equation_2d::inference::ActivationType;
    use crate::inverse::pinn::ml::wave_equation_2d::{PinnConfig2D, PinnWave2D};
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn extracted_layout_matches_every_pinn_projection_and_activation() {
        let pinn = PinnWave2D::<SequentialBackend>::new(PinnConfig2D {
            hidden_layers: vec![4, 5],
            ..PinnConfig2D::default()
        })
        .expect("non-empty hidden layer layout");

        assert_eq!(
            WaveQuantizer2D::extract_layer_sizes(&pinn),
            vec![3, 4, 5, 1]
        );
        assert_eq!(
            WaveQuantizer2D::extract_activations(&pinn),
            vec![
                ActivationType::Linear,
                ActivationType::Tanh,
                ActivationType::Linear,
            ]
        );
    }

    #[test]
    fn symmetric_quantization_preserves_signed_values_with_half_step_error() {
        let values = [-3.0_f32, -0.2, 1.0, 2.5];
        let backend = SequentialBackend;
        let tensor = Tensor::from_slice_on([values.len()], &values, &backend);
        let (quantized, scale) =
            WaveQuantizer2D::quantize_tensor(&tensor).expect("finite tensor must quantize");
        let rounding_bound =
            scale * 0.5 + values.iter().fold(0.0_f32, |m, v| m.max(v.abs())) * f32::EPSILON;

        for (&original, &encoded) in values.iter().zip(&quantized) {
            let reconstructed = encoded as f32 * scale;
            assert!((reconstructed - original).abs() <= rounding_bound);
        }
    }
}
