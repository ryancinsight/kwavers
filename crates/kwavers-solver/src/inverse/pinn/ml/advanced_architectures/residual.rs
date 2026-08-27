//! `ResidualBlock`: pre-norm residual block for PINN architectures.

use coeus_autograd::Var;
use coeus_nn::{LayerNorm, Linear, Module};
use kwavers_core::error::KwaversResult;

use crate::inverse::pinn::ml::coeus_forward::map_forward;

/// Residual block for PINN architectures with skip connections.
pub struct ResidualBlock<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    linear1: Linear<f32, B>,
    linear2: Linear<f32, B>,
    norm1: LayerNorm<f32, B>,
    norm2: LayerNorm<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> Clone for ResidualBlock<B> {
    fn clone(&self) -> Self {
        Self {
            linear1: self.linear1.clone(),
            linear2: self.linear2.clone(),
            norm1: self.norm1.clone(),
            norm2: self.norm2.clone(),
        }
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ResidualBlock<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualBlock").finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> ResidualBlock<B> {
    /// Create a new residual block.
    ///
    /// # Errors
    ///
    /// Returns [`kwavers_core::error::KwaversError::InvalidInput`] when a layer
    /// dimension is zero or the backend's weight draw fails.
    pub fn new(input_dim: usize, hidden_dim: usize) -> kwavers_core::error::KwaversResult<Self>
    where
        B: coeus_ops::RandomInitOps<f32>,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorageMut<f32>,
    {
        let linear1 = Linear::new(input_dim, hidden_dim, true).map_err(|error| {
            crate::inverse::pinn::layer_construction_failed("first residual", error)
        })?;
        let linear2 = Linear::new(hidden_dim, input_dim, true).map_err(|error| {
            crate::inverse::pinn::layer_construction_failed("second residual", error)
        })?;
        let norm1 = LayerNorm::new(hidden_dim, 1e-5);
        let norm2 = LayerNorm::new(input_dim, 1e-5);

        Ok(Self {
            linear1,
            linear2,
            norm1,
            norm2,
        })
    }

    /// Flatten all layer + norm parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.linear1.parameters();
        params.extend(self.norm1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    /// Write updated parameter values back into the block's layers.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        let n = self.linear1.parameters().len();
        self.linear1.load_parameters(&params[offset..offset + n]);
        offset += n;
        let n = self.norm1.parameters().len();
        self.norm1.load_parameters(&params[offset..offset + n]);
        offset += n;
        let n = self.linear2.parameters().len();
        self.linear2.load_parameters(&params[offset..offset + n]);
        offset += n;
        let n = self.norm2.parameters().len();
        self.norm2.load_parameters(&params[offset..offset + n]);
    }

    /// Forward pass through residual block.
    ///
    /// `out = x + LayerNorm(linear2(GELU(LayerNorm(linear1(x)))))`
    pub fn forward(&self, x: &Var<f32, B>) -> KwaversResult<Var<f32, B>> {
        let out = map_forward(self.linear1.forward(x), "PINN residual linear layer")?;
        let out = map_forward(self.norm1.forward(&out), "PINN residual normalization")?;
        let out = coeus_autograd::gelu(&out);

        let out = map_forward(self.linear2.forward(&out), "PINN residual linear layer")?;
        let out = map_forward(self.norm2.forward(&out), "PINN residual normalization")?;

        Ok(coeus_autograd::add(&out, x))
    }
}
