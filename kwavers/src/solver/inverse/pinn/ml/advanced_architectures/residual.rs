//! `ResidualBlock`: pre-norm residual block for PINN architectures.

use burn::{
    module::Module,
    nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// Residual block for PINN architectures with skip connections.
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    gelu: Gelu,
}

impl<B: Backend> ResidualBlock<B> {
    /// Create a new residual block.
    pub fn new(input_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(input_dim, hidden_dim).init(device);
        let linear2 = LinearConfig::new(hidden_dim, input_dim).init(device);
        let norm1 = LayerNormConfig::new(hidden_dim).init(device);
        let norm2 = LayerNormConfig::new(input_dim).init(device);
        let gelu = Gelu::new();

        Self {
            linear1,
            linear2,
            norm1,
            norm2,
            gelu,
        }
    }

    /// Forward pass through residual block.
    ///
    /// `out = x + LayerNorm(linear2(GELU(LayerNorm(linear1(x)))))`
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let residual = x.clone();

        let out = self.linear1.forward(x);
        let out = self.norm1.forward(out);
        let out = self.gelu.forward(out);

        let out = self.linear2.forward(out);
        let out = self.norm2.forward(out);

        out + residual
    }
}
