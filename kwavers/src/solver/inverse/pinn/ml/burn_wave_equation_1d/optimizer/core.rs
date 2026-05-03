//! Gradient descent optimizer for Burn-based 1D Wave Equation PINN.
//!
//! ## Optimization Algorithm
//!
//! Standard gradient descent parameter update: **θ = θ - α∇L**
//!
//! ## References
//!
//! - Rumelhart et al. (1986): "Learning representations by back-propagating errors"
//! - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"

use burn::{
    module::Module,
    tensor::{backend::AutodiffBackend, Bool, Int, Tensor},
};

use super::super::network::BurnPINN1DWave;

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
    /// Applies the gradient descent update rule θ = θ - α∇L to all network parameters.
    pub fn step<B: AutodiffBackend>(
        &self,
        pinn: BurnPINN1DWave<B>,
        grads: &B::Gradients,
    ) -> BurnPINN1DWave<B> {
        let mut mapper = GradientUpdateMapper1D {
            learning_rate: self.learning_rate,
            grads,
        };
        pinn.map(&mut mapper)
    }
}

/// Burn `ModuleMapper` for applying gradient descent updates.
///
/// Implements the Visitor pattern to traverse and update all parameters.
struct GradientUpdateMapper1D<'a, B: AutodiffBackend> {
    learning_rate: f32,
    grads: &'a B::Gradients,
}

impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for GradientUpdateMapper1D<'a, B> {
    /// Update float parameters using gradient descent: θ = θ - α∇L
    fn map_float<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        let is_require_grad = tensor.is_require_grad();
        let grad_opt = tensor.grad(self.grads);
        let mut inner = (*tensor).clone().inner();

        if let Some(grad) = grad_opt {
            inner = inner - grad.mul_scalar(self.learning_rate as f64);
        }

        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        burn::module::Param::from_tensor(out)
    }

    /// Pass through integer parameters unchanged.
    fn map_int<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Int>>,
    ) -> burn::module::Param<Tensor<B, D, Int>> {
        tensor
    }

    /// Pass through boolean parameters unchanged.
    fn map_bool<const D: usize>(
        &mut self,
        tensor: burn::module::Param<Tensor<B, D, Bool>>,
    ) -> burn::module::Param<Tensor<B, D, Bool>> {
        tensor
    }
}
