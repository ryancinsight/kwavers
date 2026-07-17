/// Activation Function Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Tanh activation (standard for PINNs)
    Tanh,
    /// ReLU activation (faster alternative)
    Relu,
    /// Linear activation (output layer)
    Linear,
}

/// Quantized Neural Network for Real-Time Inference
#[derive(Debug, Clone)]
pub struct QuantizedNetwork {
    /// Quantized weights for each layer, indexed by layer and weight.
    pub weights: Vec<Vec<i8>>,
    /// Quantization scales for each layer.
    pub weight_scales: Vec<f32>,
    /// Quantized biases for each layer, indexed by layer and bias.
    pub biases: Vec<Vec<i8>>,
    /// Bias quantization scales for each layer.
    pub bias_scales: Vec<f32>,
    /// Layer sizes [input_size, hidden_sizes..., output_size]
    pub layer_sizes: Vec<usize>,
    /// Activation function type per layer
    pub activations: Vec<ActivationType>,
}

impl QuantizedNetwork {
    /// Returns a per-sample upper bound on the error introduced by symmetric
    /// int8 quantization.
    ///
    /// Every quantized coefficient differs from its source value by at most
    /// half a quantization step. The bound propagates that error through each
    /// affine layer; the supported activations are all 1-Lipschitz.
    pub(crate) fn prediction_error_bounds(
        &self,
        x: &[f32],
        y: &[f32],
        t: &[f32],
    ) -> kwavers_core::error::KwaversResult<Vec<f32>> {
        self.validate_scalar_prediction_inputs(x, y, t)?;

        let mut bounds = Vec::with_capacity(x.len());
        for sample in 0..x.len() {
            let mut values = vec![x[sample], y[sample], t[sample]];
            let mut errors = vec![0.0; values.len()];

            for layer in 0..self.weights.len() {
                let input_size = self.layer_sizes[layer];
                let output_size = self.layer_sizes[layer + 1];
                let weight_scale = self.weight_scales[layer].abs();
                let bias_scale = self.bias_scales[layer].abs();
                let weight_error = weight_scale * 0.5;
                let bias_error = bias_scale * 0.5;
                let weights = &self.weights[layer];
                let biases = &self.biases[layer];

                let mut next_values = Vec::with_capacity(output_size);
                let mut next_errors = Vec::with_capacity(output_size);
                for output in 0..output_size {
                    let mut value = biases[output] as f32 * self.bias_scales[layer];
                    let mut error = bias_error;

                    for input in 0..input_size {
                        let weight =
                            weights[output * input_size + input] as f32 * self.weight_scales[layer];
                        let input_value = values[input];
                        let input_error = errors[input];

                        value = input_value.mul_add(weight, value);
                        error += input_value.abs() * weight_error
                            + weight.abs() * input_error
                            + weight_error * input_error;
                    }

                    let activated = match self.activations[layer] {
                        ActivationType::Tanh => value.tanh(),
                        ActivationType::Relu => value.max(0.0),
                        ActivationType::Linear => value,
                    };
                    next_values.push(activated);
                    next_errors.push(error);
                }

                values = next_values;
                errors = next_errors;
            }

            bounds.push(errors[0]);
        }

        Ok(bounds)
    }

    fn validate_scalar_prediction_inputs(
        &self,
        x: &[f32],
        y: &[f32],
        t: &[f32],
    ) -> kwavers_core::error::KwaversResult<()> {
        use kwavers_core::error::KwaversError;

        if x.len() != y.len() || x.len() != t.len() {
            return Err(KwaversError::InvalidInput(
                "Input coordinate arrays must have the same length".into(),
            ));
        }
        if x.iter().chain(y).chain(t).any(|value| !value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Input coordinates must be finite".into(),
            ));
        }
        if self.layer_sizes.len() != self.weights.len() + 1
            || self.biases.len() != self.weights.len()
            || self.weight_scales.len() != self.weights.len()
            || self.bias_scales.len() != self.weights.len()
            || self.activations.len() != self.weights.len()
        {
            return Err(KwaversError::InvalidInput(
                "Quantized network layer metadata is inconsistent".into(),
            ));
        }
        if self.layer_sizes.first() != Some(&3) || self.layer_sizes.last() != Some(&1) {
            return Err(KwaversError::InvalidInput(
                "Real-time wave inference requires three inputs and one output".into(),
            ));
        }

        for layer in 0..self.weights.len() {
            let input_size = self.layer_sizes[layer];
            let output_size = self.layer_sizes[layer + 1];
            let expected_weight_count = input_size.checked_mul(output_size).ok_or_else(|| {
                KwaversError::InvalidInput("Quantized layer dimensions overflow usize".into())
            })?;
            if self.weights[layer].len() != expected_weight_count
                || self.biases[layer].len() != output_size
                || !self.weight_scales[layer].is_finite()
                || !self.bias_scales[layer].is_finite()
                || self.weight_scales[layer] < 0.0
                || self.bias_scales[layer] < 0.0
            {
                return Err(KwaversError::InvalidInput(format!(
                    "Quantized layer {layer} does not match its declared dimensions"
                )));
            }
        }

        Ok(())
    }
}

/// Memory Pool for Zero-Allocation Inference
#[derive(Debug)]
pub struct WaveInferenceMemoryPool2D {
    /// Pre-allocated buffers for intermediate activations
    pub buffers: Vec<Vec<f32>>,
    /// Buffer sizes for each layer
    pub _buffer_sizes: Vec<usize>,
}

/// SIMD Processor for CPU Vectorization
#[cfg(feature = "simd")]
#[derive(Debug)]
pub struct SIMDProcessor {
    /// SIMD lanes available (typically 16 for f32x16)
    pub lanes: usize,
}

/// Neural network state for GPU inference over a pre-quantized network.
#[cfg(feature = "gpu")]
pub struct PinnNeuralNetwork<B: coeus_ops::BackendOps<f32> + Default> {
    pub weights: Vec<coeus_tensor::Tensor<f32, B>>,
    pub biases: Vec<coeus_tensor::Tensor<f32, B>>,
    pub(super) quantized_network: std::sync::Arc<QuantizedNetwork>,
}

#[cfg(feature = "gpu")]
impl<B: coeus_ops::BackendOps<f32> + Default> std::fmt::Debug for PinnNeuralNetwork<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnNeuralNetwork")
            .field("layer_count", &self.weights.len())
            .field("quantized_network", &self.quantized_network)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{ActivationType, QuantizedNetwork};

    #[test]
    fn quantization_error_bound_matches_one_layer_affine_contract() {
        let scale = 1.0 / 127.0;
        let network = QuantizedNetwork {
            weights: vec![vec![127, -127, 0]],
            weight_scales: vec![scale],
            biases: vec![vec![0]],
            bias_scales: vec![0.0],
            layer_sizes: vec![3, 1],
            activations: vec![ActivationType::Linear],
        };

        let bounds = network
            .prediction_error_bounds(&[2.0], &[-3.0], &[0.5])
            .expect("valid one-layer quantized network");
        let expected = (2.0 + 3.0 + 0.5) * scale * 0.5;
        let rounding_bound = expected.abs() * f32::EPSILON;

        assert!((bounds[0] - expected).abs() <= rounding_bound);
    }
}
