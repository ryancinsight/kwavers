//! ResNet-based PINN architectures for 1D and 2D wave equations.

use coeus_autograd::Var;
use coeus_nn::{Linear, Module};

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
pub struct ResNetPINN1D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    fourier_features: Option<FourierFeatures<B>>,
    input_proj: Linear<f32, B>,
    residual_blocks: Vec<ResidualBlock<B>>,
    output_proj: Linear<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ResNetPINN1D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResNetPINN1D")
            .field("num_blocks", &self.residual_blocks.len())
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> ResNetPINN1D<B> {
    /// Create a new ResNet-based PINN for 1D problems.
    pub fn new(config: &ResNetPINNConfig) -> Self {
        let fourier_features = if config.use_fourier_features {
            Some(FourierFeatures::new(
                config.input_dim,
                config.hidden_layers[0],
                config.fourier_scale,
            ))
        } else {
            None
        };

        let input_dim = if config.use_fourier_features {
            config.hidden_layers[0] * 2 // cos + sin features
        } else {
            config.input_dim
        };

        let input_proj = Linear::new(input_dim, config.hidden_layers[0], true);

        let mut residual_blocks = Vec::new();
        for _ in 0..config.num_blocks {
            residual_blocks.push(ResidualBlock::new(
                config.hidden_layers[0],
                config.hidden_layers[1],
            ));
        }

        let output_proj = Linear::new(config.hidden_layers[0], 1, true);

        Self {
            fourier_features,
            input_proj,
            residual_blocks,
            output_proj,
        }
    }

    /// Flatten all parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = Vec::new();
        if let Some(fourier) = &self.fourier_features {
            params.extend(fourier.parameters());
        }
        params.extend(self.input_proj.parameters());
        for block in &self.residual_blocks {
            params.extend(block.parameters());
        }
        params.extend(self.output_proj.parameters());
        params
    }

    /// Write updated parameter values back into the network's layers.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        if let Some(fourier) = &mut self.fourier_features {
            let n = fourier.parameters().len();
            fourier.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n = self.input_proj.parameters().len();
        self.input_proj.load_parameters(&params[offset..offset + n]);
        offset += n;
        for block in &mut self.residual_blocks {
            let n = block.parameters().len();
            block.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n = self.output_proj.parameters().len();
        self.output_proj
            .load_parameters(&params[offset..offset + n]);
    }

    /// Forward pass through ResNet PINN.
    pub fn forward(&self, x: &Var<f32, B>, t: &Var<f32, B>) -> Var<f32, B>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let input = coeus_autograd::cat(&[x, t], 1);

        let features = if let Some(fourier) = &self.fourier_features {
            fourier.forward(&input)
        } else {
            input
        };

        let mut out = coeus_autograd::gelu(&self.input_proj.forward(&features));

        for block in &self.residual_blocks {
            out = coeus_autograd::gelu(&block.forward(&out));
        }

        self.output_proj.forward(&out)
    }
}

/// ResNet-based PINN for 2D wave equations with residual connections.
pub struct ResNetPINN2D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    fourier_features: Option<FourierFeatures<B>>,
    input_proj: Linear<f32, B>,
    residual_blocks: Vec<ResidualBlock<B>>,
    output_proj: Linear<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ResNetPINN2D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResNetPINN2D")
            .field("num_blocks", &self.residual_blocks.len())
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> ResNetPINN2D<B> {
    /// Create a new ResNet-based PINN for 2D problems.
    pub fn new(config: &ResNetPINNConfig) -> Self {
        let fourier_features = if config.use_fourier_features {
            Some(FourierFeatures::new(
                config.input_dim,
                config.hidden_layers[0],
                config.fourier_scale,
            ))
        } else {
            None
        };

        let input_dim = if config.use_fourier_features {
            config.hidden_layers[0] * 2
        } else {
            config.input_dim
        };

        let input_proj = Linear::new(input_dim, config.hidden_layers[0], true);

        let mut residual_blocks = Vec::new();
        for _ in 0..config.num_blocks {
            residual_blocks.push(ResidualBlock::new(
                config.hidden_layers[0],
                config.hidden_layers[1],
            ));
        }

        let output_proj = Linear::new(config.hidden_layers[0], 1, true);

        Self {
            fourier_features,
            input_proj,
            residual_blocks,
            output_proj,
        }
    }

    /// Flatten all parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = Vec::new();
        if let Some(fourier) = &self.fourier_features {
            params.extend(fourier.parameters());
        }
        params.extend(self.input_proj.parameters());
        for block in &self.residual_blocks {
            params.extend(block.parameters());
        }
        params.extend(self.output_proj.parameters());
        params
    }

    /// Write updated parameter values back into the network's layers.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        if let Some(fourier) = &mut self.fourier_features {
            let n = fourier.parameters().len();
            fourier.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n = self.input_proj.parameters().len();
        self.input_proj.load_parameters(&params[offset..offset + n]);
        offset += n;
        for block in &mut self.residual_blocks {
            let n = block.parameters().len();
            block.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n = self.output_proj.parameters().len();
        self.output_proj
            .load_parameters(&params[offset..offset + n]);
    }

    /// Forward pass through ResNet PINN.
    pub fn forward(&self, x: &Var<f32, B>, y: &Var<f32, B>, t: &Var<f32, B>) -> Var<f32, B>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let input = coeus_autograd::cat(&[x, y, t], 1);

        let features = if let Some(fourier) = &self.fourier_features {
            fourier.forward(&input)
        } else {
            input
        };

        let mut out = coeus_autograd::gelu(&self.input_proj.forward(&features));

        for block in &self.residual_blocks {
            out = coeus_autograd::gelu(&block.forward(&out));
        }

        self.output_proj.forward(&out)
    }
}
