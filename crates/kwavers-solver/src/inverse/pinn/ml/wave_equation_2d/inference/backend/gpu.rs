#[cfg(feature = "gpu")]
use super::super::types::PinnNeuralNetwork;
#[cfg(feature = "gpu")]
use super::super::types::QuantizedNetwork;
#[cfg(feature = "gpu")]
use coeus_autograd::Var;
#[cfg(feature = "gpu")]
use kwavers_core::error::{KwaversError, KwaversResult};

#[cfg(feature = "gpu")]
impl<B: coeus_ops::BackendOps<f32> + Default> PinnNeuralNetwork<B> {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(network: &QuantizedNetwork) -> KwaversResult<Self> {
        let backend = B::default();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for (i, layer_weights) in network.weights.iter().enumerate() {
            let scale = network.weight_scales[i];
            let f32_weights: Vec<f32> = layer_weights.iter().map(|&w| w as f32 * scale).collect();

            let input_dim = if i == 0 { 3 } else { network.layer_sizes[i] };
            let output_dim = network.layer_sizes[i + 1];

            let weight_tensor = coeus_tensor::Tensor::from_slice_on(
                vec![input_dim, output_dim],
                &f32_weights,
                &backend,
            );
            weights.push(weight_tensor);
        }

        for (i, layer_biases) in network.biases.iter().enumerate() {
            let scale = network.bias_scales[i];
            let f32_biases: Vec<f32> = layer_biases.iter().map(|&b| b as f32 * scale).collect();

            let output_dim = network.layer_sizes[i + 1];
            let bias_tensor =
                coeus_tensor::Tensor::from_slice_on(vec![output_dim], &f32_biases, &backend);
            biases.push(bias_tensor);
        }

        let activation_str = if !network.activations.is_empty() {
            format!("{:?}", network.activations[0]).to_lowercase()
        } else {
            "tanh".to_string()
        };

        Ok(Self {
            weights,
            biases,
            activation: activation_str,
        })
    }
    /// Predict.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn predict(&self, x: &[f32], y: &[f32], t: &[f32]) -> KwaversResult<(Vec<f32>, Vec<f32>)> {
        let batch_size = (x.shape()[0] * x.shape()[1] * x.shape()[2]);
        if (y.shape()[0] * y.shape()[1] * y.shape()[2]) != batch_size || (t.shape()[0] * t.shape()[1] * t.shape()[2]) != batch_size {
            return Err(KwaversError::InvalidInput(
                "Input coordinate arrays must have the same length".into(),
            ));
        }

        let backend = B::default();
        let mut input_data = Vec::with_capacity(batch_size * 3);
        for i in 0..batch_size {
            input_data.push(x[i]);
            input_data.push(y[i]);
            input_data.push(t[i]);
        }

        let mut input = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 3], &input_data, &backend),
            false,
        );

        for (layer_idx, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            let weight_var = Var::new(weight.clone(), false);
            let bias_var = Var::new(bias.clone(), false);
            let out = coeus_autograd::matmul(&input, &weight_var);
            input = coeus_autograd::add(&out, &bias_var);

            if layer_idx < (self.weights.shape()[0] * self.weights.shape()[1] * self.weights.shape()[2]) - 1 {
                input = match self.activation.as_str() {
                    "relu" => coeus_autograd::relu(&input),
                    "sigmoid" => coeus_autograd::sigmoid(&input),
                    _ => coeus_autograd::tanh(&input),
                };
            }
        }

        let predictions: Vec<f32> = input.tensor.as_slice().to_vec();
        let uncertainties = vec![0.1; batch_size];

        Ok((predictions, uncertainties))
    }
}
