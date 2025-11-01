//! Advanced Neural Architectures for Physics-Informed Neural Networks (PINNs)
//!
//! This module implements state-of-the-art neural architectures specifically designed
//! for physics-informed neural networks. These architectures address common PINN
//! convergence issues and improve accuracy for solving partial differential equations.
//!
//! ## Architectures Implemented
//!
//! - **ResNet PINNs**: Residual connections to enable deeper networks and better gradient flow
//! - **Fourier Features**: Frequency-domain embeddings for better representation of oscillatory physics
//!
//! ## References
//!
//! - Wang et al. (2021): "When and why PINNs fail to train: A neural tangent kernel perspective"
//! - Wang et al. (2022): "On the eigenvector bias of Fourier feature networks"

//! Fourier feature embedding for improved frequency representation in PINNs
//!
//! Fourier features provide better representation of oscillatory physics by mapping
//! input coordinates to higher-frequency sinusoidal functions.

use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig, LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

/// Fourier feature embedding for improved frequency representation
#[derive(Clone)]
pub struct FourierFeatures<B: Backend> {
    /// Learned frequency parameters
    frequencies: Tensor<B, 2>,
    /// Feature scaling factor
    scale: f32,
}

impl<B: Backend> FourierFeatures<B> {
    /// Create Fourier feature embedding
    pub fn new(input_dim: usize, num_features: usize, scale: f32, device: &B::Device) -> Self {
        // Initialize frequencies randomly
        let freq_data: Vec<f32> = (0..input_dim * num_features)
            .map(|_| rand::random::<f32>() * scale)
            .collect();

        let frequencies = Tensor::<B, 2>::from_floats(freq_data.as_slice(), device)
            .reshape([input_dim as i64, num_features as i64]);

        Self {
            frequencies,
            scale,
        }
    }

    /// Apply Fourier feature transformation
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // x: [batch_size, input_dim]
        // frequencies: [input_dim, num_features]

        // Compute Fourier features: [cos(2π * x * f), sin(2π * x * f)]
        let scaled_x = x * self.scale;
        let features = scaled_x.matmul(self.frequencies);

        // Apply sinusoidal transformations
        let cos_features = features.cos();
        let sin_features = features.sin();

        // Concatenate cos and sin features
        Tensor::cat(vec![cos_features, sin_features], 1)
    }
}

/// Residual block for PINN architectures with skip connections
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    /// First linear layer
    linear1: Linear<B>,
    /// Second linear layer
    linear2: Linear<B>,
    /// Layer normalization
    norm1: LayerNorm<B>,
    /// Second layer normalization
    norm2: LayerNorm<B>,
    /// GELU activation
    gelu: Gelu,
}

impl<B: Backend> ResidualBlock<B> {
    /// Create a new residual block
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

    /// Forward pass through residual block
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let residual = x.clone();

        // First transformation
        let out = self.linear1.forward(x);
        let out = self.norm1.forward(out);
        let out = self.gelu.forward(out);

        // Second transformation
        let out = self.linear2.forward(out);
        let out = self.norm2.forward(out);

        // Residual connection
        out + residual
    }
}

/// Configuration for ResNet-based PINN architectures
#[derive(Debug, Clone)]
pub struct ResNetPINNConfig {
    /// Number of input features (spatial + temporal dimensions)
    pub input_dim: usize,
    /// Hidden layer sizes for each residual block
    pub hidden_layers: Vec<usize>,
    /// Number of residual blocks
    pub num_blocks: usize,
    /// Whether to use Fourier feature embeddings
    pub use_fourier_features: bool,
    /// Fourier feature scale parameter
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

/// ResNet-based PINN for 1D wave equations with residual connections
#[derive(Module, Debug)]
pub struct ResNetPINN1D<B: Backend> {
    /// Fourier feature embedding (optional)
    fourier_features: Option<FourierFeatures<B>>,
    /// Input projection layer
    input_proj: Linear<B>,
    /// Residual blocks
    residual_blocks: Vec<ResidualBlock<B>>,
    /// Output projection layer
    output_proj: Linear<B>,
    /// GELU activation
    gelu: Gelu,
}

impl<B: Backend> ResNetPINN1D<B> {
    /// Create a new ResNet-based PINN for 1D problems
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

    /// Forward pass through ResNet PINN
    pub fn forward(&self, x: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate spatial and temporal inputs
        let input = Tensor::cat(vec![x, t], 1);

        // Apply Fourier features if enabled
        let features = if let Some(ref fourier) = self.fourier_features {
            fourier.forward(input)
        } else {
            input
        };

        // Input projection
        let mut out = self.input_proj.forward(features);
        out = self.gelu.forward(out);

        // Apply residual blocks
        for block in &self.residual_blocks {
            out = block.forward(out);
            out = self.gelu.forward(out);
        }

        // Output projection
        self.output_proj.forward(out)
    }
}

/// ResNet-based PINN for 2D wave equations with residual connections
#[derive(Module, Debug)]
pub struct ResNetPINN2D<B: Backend> {
    /// Fourier feature embedding (optional)
    fourier_features: Option<FourierFeatures<B>>,
    /// Input projection layer
    input_proj: Linear<B>,
    /// Residual blocks
    residual_blocks: Vec<ResidualBlock<B>>,
    /// Output projection layer
    output_proj: Linear<B>,
    /// GELU activation
    gelu: Gelu,
}

impl<B: Backend> ResNetPINN2D<B> {
    /// Create a new ResNet-based PINN for 2D problems
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

    /// Forward pass through ResNet PINN
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate spatial and temporal inputs
        let input = Tensor::cat(vec![x, y, t], 1);

        // Apply Fourier features if enabled
        let features = if let Some(ref fourier) = self.fourier_features {
            fourier.forward(input)
        } else {
            input
        };

        // Input projection
        let mut out = self.input_proj.forward(features);
        out = self.gelu.forward(out);

        // Apply residual blocks
        for block in &self.residual_blocks {
            out = block.forward(out);
            out = self.gelu.forward(out);
        }

        // Output projection
        self.output_proj.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_fourier_features() {
        let device = Default::default();
        let fourier = FourierFeatures::<TestBackend>::new(2, 10, 1.0, &device);

        let input = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.3]], &device);
        let output = fourier.forward(input);

        // Should have 20 features (10 cos + 10 sin)
        assert_eq!(output.dims(), [1, 20]);
    }

    #[test]
    fn test_residual_block() {
        let device = Default::default();
        let block = ResidualBlock::<TestBackend>::new(10, 20, &device);

        let input = Tensor::<TestBackend, 2>::from_floats(vec![vec![0.1; 10]], &device);
        let output = block.forward(input);

        // Should maintain input dimension
        assert_eq!(output.dims(), [1, 10]);
    }

    #[test]
    fn test_resnet_pinn_1d() {
        let device = Default::default();
        let config = ResNetPINNConfig {
            input_dim: 2,
            hidden_layers: vec![32, 64],
            num_blocks: 2,
            use_fourier_features: true,
            fourier_scale: 5.0,
        };

        let pinn = ResNetPINN1D::<TestBackend>::new(&config, &device);

        let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
        let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
        let output = pinn.forward(x, t);

        assert_eq!(output.dims(), [1, 1]);
    }

    #[test]
    fn test_resnet_pinn_2d() {
        let device = Default::default();
        let config = ResNetPINNConfig {
            input_dim: 3,
            hidden_layers: vec![32, 64],
            num_blocks: 2,
            use_fourier_features: true,
            fourier_scale: 5.0,
        };

        let pinn = ResNetPINN2D::<TestBackend>::new(&config, &device);

        let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
        let y = Tensor::<TestBackend, 2>::from_floats([[0.3]], &device);
        let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
        let output = pinn.forward(x, y, t);

        assert_eq!(output.dims(), [1, 1]);
    }
}
