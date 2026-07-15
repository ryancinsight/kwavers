#[cfg(feature = "gpu")]
use super::super::types::PinnNeuralNetwork;
#[cfg(feature = "gpu")]
use super::super::types::QuantizedNetwork;
#[cfg(feature = "gpu")]
use coeus_autograd::Var;
#[cfg(feature = "gpu")]
use kwavers_core::error::KwaversResult;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
impl<B: coeus_ops::BackendOps<f32> + Default> PinnNeuralNetwork<B> {
    /// New.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn new(network: Arc<QuantizedNetwork>) -> KwaversResult<Self> {
        network.prediction_error_bounds(&[], &[], &[])?;
        let backend = B::default();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for (i, layer_weights) in network.weights.iter().enumerate() {
            let scale = network.weight_scales[i];
            let f32_weights: Vec<f32> = layer_weights.iter().map(|&w| w as f32 * scale).collect();

            let input_dim = network.layer_sizes[i];
            let output_dim = network.layer_sizes[i + 1];

            let weight_tensor = coeus_tensor::Tensor::from_slice_on(
                vec![output_dim, input_dim],
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

        Ok(Self {
            weights,
            biases,
            quantized_network: network,
        })
    }
    /// Predict.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn predict(&self, x: &[f32], y: &[f32], t: &[f32]) -> KwaversResult<(Vec<f32>, Vec<f32>)> {
        let batch_size = x.len();
        let uncertainties = self.quantized_network.prediction_error_bounds(x, y, t)?;

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
            let transposed_weight = coeus_autograd::transpose_2d(&weight_var);
            let out = coeus_autograd::matmul(&input, &transposed_weight);
            input = coeus_autograd::add(&out, &bias_var);

            input = match self.quantized_network.activations[layer_idx] {
                super::super::types::ActivationType::Tanh => coeus_autograd::tanh(&input),
                super::super::types::ActivationType::Relu => coeus_autograd::relu(&input),
                super::super::types::ActivationType::Linear => input,
            };
        }

        let mut predictions = vec![0.0; batch_size];
        backend.copy_to_host(input.tensor.storage(), &mut predictions);

        Ok((predictions, uncertainties))
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{PinnNeuralNetwork, QuantizedNetwork};
    use crate::inverse::pinn::ml::wave_equation_2d::inference::ActivationType;
    use coeus_core::SequentialBackend;
    use std::sync::Arc;

    #[test]
    fn provider_predict_preserves_linear_weight_orientation_and_error_bound() {
        let scale = 1.0 / 127.0;
        let network = Arc::new(QuantizedNetwork {
            weights: vec![vec![127, -127, 0]],
            weight_scales: vec![scale],
            biases: vec![vec![0]],
            bias_scales: vec![0.0],
            layer_sizes: vec![3, 1],
            activations: vec![ActivationType::Linear],
        });
        let predictor =
            PinnNeuralNetwork::<SequentialBackend>::new(network).expect("valid quantized network");

        let (predictions, bounds) = predictor
            .predict(&[2.0], &[-3.0], &[0.5])
            .expect("valid coordinate batch");
        let expected_bound = (2.0 + 3.0 + 0.5) * scale * 0.5;

        assert_eq!(predictions, vec![5.0]);
        assert!((bounds[0] - expected_bound).abs() <= expected_bound * f32::EPSILON);
    }
}
