#[cfg(feature = "gpu")]
use super::super::types::BurnNeuralNetwork;
#[cfg(feature = "gpu")]
use super::super::types::QuantizedNetwork;
#[cfg(feature = "gpu")]
use crate::core::error::{KwaversError, KwaversResult};
#[cfg(feature = "gpu")]
use burn::tensor::activation::{relu, sigmoid, tanh};
#[cfg(feature = "gpu")]
use burn::tensor::{backend::Backend, Tensor, TensorData};

#[cfg(feature = "gpu")]
impl<B: Backend> BurnNeuralNetwork<B> {
    pub fn new(network: &QuantizedNetwork, device: &B::Device) -> KwaversResult<Self> {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for (i, layer_weights) in network.weights.iter().enumerate() {
            let scale = network.weight_scales[i];
            let f32_weights: Vec<f32> = layer_weights.iter().map(|&w| w as f32 * scale).collect();

            let input_dim = if i == 0 { 3 } else { network.layer_sizes[i] };
            let output_dim = network.layer_sizes[i + 1];

            let data = TensorData::new(f32_weights, [input_dim, output_dim]);
            let weight_tensor = Tensor::<B, 2>::from_data(data, device);
            weights.push(weight_tensor);
        }

        for (i, layer_biases) in network.biases.iter().enumerate() {
            let scale = network.bias_scales[i];
            let f32_biases: Vec<f32> = layer_biases.iter().map(|&b| b as f32 * scale).collect();

            let output_dim = network.layer_sizes[i + 1];
            let data = TensorData::new(f32_biases, [output_dim]);
            let bias_tensor = Tensor::<B, 1>::from_data(data, device);
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

    pub fn predict(&self, x: &[f32], y: &[f32], t: &[f32]) -> KwaversResult<(Vec<f32>, Vec<f32>)> {
        let batch_size = x.len();
        if y.len() != batch_size || t.len() != batch_size {
            return Err(KwaversError::InvalidInput(
                "Input coordinate arrays must have the same length".into(),
            ));
        }

        let mut input_data = Vec::with_capacity(batch_size * 3);
        for i in 0..batch_size {
            input_data.push(x[i]);
            input_data.push(y[i]);
            input_data.push(t[i]);
        }

        let device = self.weights[0].device();
        let data = TensorData::new(input_data, [batch_size, 3]);
        let mut input = Tensor::<B, 2>::from_data(data, &device);

        for (layer_idx, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            input = input.matmul(weight.clone()) + bias.clone().unsqueeze();

            if layer_idx < self.weights.len() - 1 {
                match self.activation.as_str() {
                    "relu" => input = relu(input),
                    "sigmoid" => input = sigmoid(input),
                    "tanh" => input = tanh(input),
                    _ => input = tanh(input),
                }
            }
        }

        let predictions: Vec<f32> = input.into_data().to_vec().unwrap_or_default();
        let uncertainties = vec![0.1; batch_size];

        Ok((predictions, uncertainties))
    }
}
