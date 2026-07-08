//! Gradient descent optimizer for Coeus-backed 1D Wave Equation PINN.
//!
//! ## Optimization Algorithm
//!
//! Standard gradient descent parameter update: **θ = θ - α∇L**
//!
//! ## References
//!
//! - Rumelhart et al. (1986): "Learning representations by back-propagating errors"
//! - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"

use coeus_optim::{Optimizer as CoeusOptimizer, SGD};

use super::super::network::PinnWave1D;

/// Simple gradient descent optimizer for PINN training.
///
/// Implements the standard gradient descent update rule: θ = θ - α∇L
#[derive(Debug, Clone)]
pub struct SimpleOptimizer {
    /// Learning rate (α) for gradient descent.
    learning_rate: f32,
}

impl SimpleOptimizer {
    /// Create a new gradient descent optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Update network parameters using gradient descent.
    ///
    /// Applies the gradient descent update rule θ = θ - α∇L to all network
    /// parameters. `SGD::step` mutates its own owned copy of the parameters
    /// (`coeus_tensor::Tensor`'s storage is copy-on-write, so a clone taken
    /// via `parameters()` detaches from the network on first mutation);
    /// `load_parameters` writes the updated values back into `pinn`'s layers.
    pub fn step<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        mut pinn: PinnWave1D<B>,
    ) -> PinnWave1D<B>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let mut opt = SGD::new(pinn.parameters(), self.learning_rate, 0.0);
        opt.step();
        pinn.load_parameters(&opt.params);
        pinn
    }
}
