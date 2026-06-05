//! ResNet-based PINN architectures for 1D and 2D wave equations.

use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use super::fourier::FourierFeatures;
use super::residual::ResidualBlock;

/// Configuration for ResNet-based PINN architectures.
#[derive(Debug, Clone)]
pub struct ResNetPINNConfig {
    /// Number of input features (spatial + temporal dimensions).
    pub input_dim: usize,
    /// Hidden layer sizes for each residual block.
    pub hidden_layers: Vec<usize>,
    /// Number of residual blocks.
    pub num_blocks: usize,
    /// Whether to use Fourier feature embeddings.
    pub use_fourier_features: bool,
    /// Fourier feature scale parameter.
    pub fourier_scale: f32,
}

impl Default for ResNetPINNConfig {
    fn default() -> Self {
        Self {
            input_dim: 2, // x, t for 1D problems
            hidden_layers: vec![100, 100, 100],
            num_blocks: 3,
            use_fourier_features: true,
            fourier_scale: 10.0,
        }
    }
}

/// ResNet-based PINN for 1D wave equations with residual connections.
#[derive(Module, Debug)]
pub struct ResNetPINN1D<B: Backend> {
    fourier_features: Option<FourierFeatures<B>>,
    input_proj: Linear<B>,
    residual_blocks: Vec<ResidualBlock<B>>,
    output_proj: Linear<B>,
    gelu: Gelu,
}

impl<B: Backend> ResNetPINN1D<B> {
    /// Create a new ResNet-based PINN for 1D problems.
    pub fn new(config: &ResNetPINNConfig, device: &B::Device) -> Self {
        let fourier_features = if config.use_fourier_features {
            Some(FourierFeatures::new(
                config.input_dim,
                config.hidden_layers[0],
                config.fourier_scale,
                device,
            ))
        } else {
            None
        };

        let input_dim = if config.use_fourier_features {
            config.hidden_layers[0] * 2 // cos + sin features
        } else {
            config.input_dim
        };

        let input_proj = LinearConfig::new(input_dim, config.hidden_layers[0]).init(device);

        let mut residual_blocks = Vec::new();
        for _ in 0..config.num_blocks {
            residual_blocks.push(ResidualBlock::new(
                config.hidden_layers[0],
                config.hidden_layers[1],
                device,
            ));
        }

        let output_proj = LinearConfig::new(config.hidden_layers[0], 1).init(device);
        let gelu = Gelu::new();

        Self {
            fourier_features,
            input_proj,
            residual_blocks,
            output_proj,
            gelu,
        }
    }

    /// Forward pass through ResNet PINN.
    pub fn forward(&self, x: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![x, t], 1);

        let features = if let Some(ref fourier) = self.fourier_features {
            fourier.forward(input)
        } else {
            input
        };

        let mut out = self.input_proj.forward(features);
        out = self.gelu.forward(out);

        for block in &self.residual_blocks {
            out = block.forward(out);
            out = self.gelu.forward(out);
        }

        self.output_proj.forward(out)
    }
}

/// ResNet-based PINN for 2D wave equations with residual connections.
#[derive(Module, Debug)]
pub struct ResNetPINN2D<B: Backend> {
    fourier_features: Option<FourierFeatures<B>>,
    input_proj: Linear<B>,
    residual_blocks: Vec<ResidualBlock<B>>,
    output_proj: Linear<B>,
    gelu: Gelu,
}

impl<B: Backend> ResNetPINN2D<B> {
    /// Create a new ResNet-based PINN for 2D problems.
    pub fn new(config: &ResNetPINNConfig, device: &B::Device) -> Self {
        let fourier_features = if config.use_fourier_features {
            Some(FourierFeatures::new(
                config.input_dim,
                config.hidden_layers[0],
                config.fourier_scale,
                device,
            ))
        } else {
            None
        };

        let input_dim = if config.use_fourier_features {
            config.hidden_layers[0] * 2
        } else {
            config.input_dim
        };

        let input_proj = LinearConfig::new(input_dim, config.hidden_layers[0]).init(device);

        let mut residual_blocks = Vec::new();
        for _ in 0..config.num_blocks {
            residual_blocks.push(ResidualBlock::new(
                config.hidden_layers[0],
                config.hidden_layers[1],
                device,
            ));
        }

        let output_proj = LinearConfig::new(config.hidden_layers[0], 1).init(device);
        let gelu = Gelu::new();

        Self {
            fourier_features,
            input_proj,
            residual_blocks,
            output_proj,
            gelu,
        }
    }

    /// Forward pass through ResNet PINN.
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![x, y, t], 1);

        let features = if let Some(ref fourier) = self.fourier_features {
            fourier.forward(input)
        } else {
            input
        };

        let mut out = self.input_proj.forward(features);
        out = self.gelu.forward(out);

        for block in &self.residual_blocks {
            out = block.forward(out);
            out = self.gelu.forward(out);
        }

        self.output_proj.forward(out)
    }
}
