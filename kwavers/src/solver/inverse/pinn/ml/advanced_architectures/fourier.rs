//! `FourierFeatures`: frequency-domain input embedding for PINNs.

use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

/// Fourier feature embedding for improved frequency representation.
///
/// Fourier features provide better representation of oscillatory physics by mapping
/// input coordinates to higher-frequency sinusoidal functions.
#[derive(Debug, Module)]
pub struct FourierFeatures<B: Backend> {
    /// Learned frequency parameters.
    frequencies: Param<Tensor<B, 2>>,
    /// Feature scaling factor.
    #[module(ignore)]
    scale: f32,
}

impl<B: Backend> FourierFeatures<B> {
    /// Create Fourier feature embedding.
    pub fn new(input_dim: usize, num_features: usize, scale: f32, device: &B::Device) -> Self {
        let freq_data: Vec<f32> = (0..input_dim * num_features)
            .map(|_| rand::random::<f32>() * scale)
            .collect();

        let frequencies = Tensor::<B, 1>::from_floats(freq_data.as_slice(), device)
            .reshape([input_dim, num_features]);

        Self {
            frequencies: Param::from_tensor(frequencies),
            scale,
        }
    }

    /// Apply Fourier feature transformation.
    ///
    /// Maps `x` ∈ ℝ^{batch × input_dim} to
    /// `[cos(scale·x·f), sin(scale·x·f)]` ∈ ℝ^{batch × 2·num_features}.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let scaled_x = x * self.scale;
        let features = scaled_x.matmul(self.frequencies.val());

        let cos_features = features.clone().cos();
        let sin_features = features.sin();

        Tensor::cat(vec![cos_features, sin_features], 1)
    }
}
