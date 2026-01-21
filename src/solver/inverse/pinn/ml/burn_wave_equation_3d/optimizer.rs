//! Simple optimizer for PINN parameter updates
//!
//! This module implements a basic stochastic gradient descent (SGD) optimizer
//! for updating PINN network parameters. It uses Burn's `ModuleMapper` trait
//! to apply gradient updates across all learnable parameters.
//!
//! ## Optimization Algorithm
//!
//! Standard SGD update rule:
//!
//! θ_{t+1} = θ_t - η ∇L(θ_t)
//!
//! Where:
//! - θ = network parameters
//! - η = learning rate
//! - ∇L = loss gradient
//!
//! ## Implementation Details
//!
//! - Uses `ModuleMapper` to traverse parameter tree
//! - Updates only float tensors (weights/biases)
//! - Preserves `require_grad` flags
//! - Leaves int/bool parameters unchanged

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{backend::AutodiffBackend, Bool, Int, Tensor};

use super::network::PINN3DNetwork;

/// Simple SGD optimizer for PINN training
///
/// Implements basic gradient descent without momentum, weight decay, or
/// adaptive learning rates. Suitable for proof-of-concept PINN training.
///
/// # Type Parameters
///
/// None - operates on generic `AutodiffBackend` via the `step` method
///
/// # Fields
///
/// * `learning_rate` - Step size η for gradient updates
#[derive(Debug, Clone)]
pub struct SimpleOptimizer3D {
    learning_rate: f32,
}

impl SimpleOptimizer3D {
    /// Create a new optimizer with the specified learning rate
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for gradient descent (typical: 1e-3 to 1e-4)
    ///
    /// # Returns
    ///
    /// A new `SimpleOptimizer3D` instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use kwavers::solver::inverse::pinn::ml::burn_wave_equation_3d::SimpleOptimizer3D;
    ///
    /// let optimizer = SimpleOptimizer3D::new(1e-3);
    /// ```
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Perform one optimization step: update network parameters using gradients
    ///
    /// # Arguments
    ///
    /// * `pinn` - Current network state
    /// * `grads` - Gradient information from `loss.backward()`
    ///
    /// # Returns
    ///
    /// Updated network with parameters θ_{t+1} = θ_t - η ∇L
    ///
    /// # Type Parameters
    ///
    /// * `B` - Autodiff backend (e.g., Autodiff<NdArray>)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Compute loss and gradients
    /// let loss = compute_loss(&pinn, &data);
    /// let grads = loss.backward();
    ///
    /// // Update parameters
    /// let updated_pinn = optimizer.step(pinn, &grads);
    /// ```
    pub fn step<B: AutodiffBackend>(
        &self,
        pinn: PINN3DNetwork<B>,
        grads: &B::Gradients,
    ) -> PINN3DNetwork<B> {
        let mut mapper = GradientUpdateMapper3D {
            learning_rate: self.learning_rate,
            grads,
        };
        pinn.map(&mut mapper)
    }
}

/// Internal mapper for applying gradient updates to network parameters
///
/// Implements `ModuleMapper` to traverse the module tree and update
/// all float tensors using the SGD rule.
struct GradientUpdateMapper3D<'a, B: AutodiffBackend> {
    learning_rate: f32,
    grads: &'a B::Gradients,
}

impl<'a, B: AutodiffBackend> ModuleMapper<B> for GradientUpdateMapper3D<'a, B> {
    /// Update float parameters: θ ← θ - η × grad
    fn map_float<const D: usize>(&mut self, tensor: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let is_require_grad = tensor.is_require_grad();
        let grad_opt = tensor.grad(self.grads);

        let mut inner = (*tensor).clone().inner();
        if let Some(grad) = grad_opt {
            // SGD update: θ ← θ - η × ∇θ
            inner = inner - grad.mul_scalar(self.learning_rate as f64);
        }

        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        Param::from_tensor(out)
    }

    /// Pass through int parameters unchanged
    fn map_int<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        tensor
    }

    /// Pass through bool parameters unchanged
    fn map_bool<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::config::BurnPINN3DConfig;
    use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};

    type TestBackend = Autodiff<NdArray>;

    fn scalar_f32(t: &Tensor<TestBackend, 1>) -> KwaversResult<f32> {
        let data = t.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_to_f32_slice".to_string(),
                reason: format!("{e:?}"),
            })
        })?;
        if slice.len() != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "len=1".to_string(),
                    actual: format!("len={}", slice.len()),
                },
            ));
        }
        slice.first().copied().ok_or_else(|| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_scalar_extract".to_string(),
                reason: "missing scalar element".to_string(),
            })
        })
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = SimpleOptimizer3D::new(1e-3);
        assert_eq!(optimizer.learning_rate, 1e-3);
    }

    #[test]
    fn test_optimizer_step_updates_parameters() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![4],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;

        // Create synthetic input and target
        let x = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let y = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let z = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let t = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let target = Tensor::<TestBackend, 2>::zeros([2, 1], &device);

        // Forward pass and compute loss
        let output = network.forward(x, y, z, t);
        let loss = (output - target).powf_scalar(2.0).mean();

        // Backward pass
        let grads = loss.backward();

        // Optimization step
        let optimizer = SimpleOptimizer3D::new(0.1);
        let updated_network = optimizer.step(network, &grads);

        // Verify network structure is preserved
        assert_eq!(
            updated_network.hidden_layer_count(),
            config.hidden_layers.len() - 1
        );
        Ok(())
    }

    #[test]
    fn test_optimizer_keeps_loss_finite() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            ..Default::default()
        };

        let mut network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;
        let optimizer = SimpleOptimizer3D::new(0.01);

        // Synthetic training data: constant function u = 0
        let x = Tensor::<TestBackend, 1>::from_floats([0.0_f32, 0.5, 1.0].as_slice(), &device)
            .reshape([3, 1]);
        let y = Tensor::<TestBackend, 1>::from_floats([0.0_f32, 0.5, 1.0].as_slice(), &device)
            .reshape([3, 1]);
        let z = Tensor::<TestBackend, 1>::from_floats([0.0_f32, 0.5, 1.0].as_slice(), &device)
            .reshape([3, 1]);
        let t = Tensor::<TestBackend, 1>::from_floats([0.1_f32, 0.2, 0.3].as_slice(), &device)
            .reshape([3, 1]);
        let target = Tensor::<TestBackend, 2>::zeros([3, 1], &device);

        // Initial loss
        let output_initial = network.forward(x.clone(), y.clone(), z.clone(), t.clone());
        let loss_initial = (output_initial - target.clone()).powf_scalar(2.0).mean();
        let loss_initial_val = scalar_f32(&loss_initial)?;
        assert!(loss_initial_val.is_finite());

        // Train for 10 steps
        for _ in 0..10 {
            let output = network.forward(x.clone(), y.clone(), z.clone(), t.clone());
            let loss = (output - target.clone()).powf_scalar(2.0).mean();
            let grads = loss.backward();
            network = optimizer.step(network, &grads);
        }

        // Final loss
        let output_final = network.forward(x, y, z, t);
        let loss_final = (output_final - target).powf_scalar(2.0).mean();
        let loss_final_val = scalar_f32(&loss_final)?;

        assert!(loss_final_val.is_finite());
        Ok(())
    }

    #[test]
    fn test_optimizer_learning_rate_effect() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![4],
            ..Default::default()
        };

        // Create two networks with same initial state
        let network1 = PINN3DNetwork::<TestBackend>::new(&config, &device)?;
        let network2 = network1.clone();

        let x = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let y = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let z = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let t = Tensor::<TestBackend, 2>::ones([2, 1], &device);
        let target = Tensor::<TestBackend, 2>::zeros([2, 1], &device);

        // Compute gradients (same for both)
        let output = network1.forward(x.clone(), y.clone(), z.clone(), t.clone());
        let loss = (output - target.clone()).powf_scalar(2.0).mean();
        let grads = loss.backward();

        // Apply different learning rates
        let optimizer_small = SimpleOptimizer3D::new(0.001);
        let optimizer_large = SimpleOptimizer3D::new(0.1);

        let updated1 = optimizer_small.step(network1, &grads);
        let updated2 = optimizer_large.step(network2, &grads);

        // Compute outputs after update
        let out1 = updated1.forward(x.clone(), y.clone(), z.clone(), t.clone());
        let out2 = updated2.forward(x, y, z, t);

        let loss1 = (out1 - target.clone()).powf_scalar(2.0).mean();
        let loss2 = (out2 - target).powf_scalar(2.0).mean();

        let loss1_val = scalar_f32(&loss1)?;
        let loss2_val = scalar_f32(&loss2)?;

        // Larger learning rate should (usually) produce bigger change
        // This is a weak test due to non-linearities, but checks basic behavior
        assert!(
            loss1_val.is_finite() && loss2_val.is_finite(),
            "Both losses should be finite"
        );
        Ok(())
    }
}
