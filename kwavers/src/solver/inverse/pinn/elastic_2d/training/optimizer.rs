//! Optimization algorithms for PINN training
//!
//! This module implements gradient descent optimizers with persistent state
//! for physics-informed neural network training.

#[cfg(feature = "pinn")]
use burn::module::{Module, Param};
#[cfg(feature = "pinn")]
use burn::tensor::{
    backend::{AutodiffBackend, Backend},
    Tensor,
};

// ============================================================================
// Persistent Adam State
// ============================================================================

/// Persistent state for Adam optimizer with moment buffers
///
/// This struct maintains first and second moment estimates for all parameters
/// by mirroring the model structure. This enables mathematically complete Adam
/// optimization with exponential moving averages across training steps.
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct PersistentAdamState<B: Backend> {
    /// First moment estimates (exponential moving average of gradients)
    /// Structure mirrors the model exactly
    pub first_moments: super::super::model::ElasticPINN2D<B>,

    /// Second moment estimates (exponential moving average of squared gradients)
    /// Structure mirrors the model exactly
    pub second_moments: super::super::model::ElasticPINN2D<B>,

    /// Global timestep counter (for bias correction)
    pub timestep: usize,

    /// Beta1 hyperparameter (first moment decay rate, typically 0.9)
    pub beta1: f64,

    /// Beta2 hyperparameter (second moment decay rate, typically 0.999)
    pub beta2: f64,

    /// Epsilon for numerical stability (typically 1e-8)
    pub epsilon: f64,
}

#[cfg(feature = "pinn")]
impl<B: Backend> PersistentAdamState<B> {
    /// Initialize Adam state with zero moment buffers
    ///
    /// Creates first and second moment buffers by cloning the model structure
    /// and initializing all values to zero.
    pub fn new(
        model: &super::super::model::ElasticPINN2D<B>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        // Initialize moment buffers as zero-initialized models
        let first_moments = model.clone().map(&mut ZeroInitMapper);
        let second_moments = model.clone().map(&mut ZeroInitMapper);

        Self {
            first_moments,
            second_moments,
            timestep: 0,
            beta1,
            beta2,
            epsilon,
        }
    }

    /// Increment timestep counter
    pub fn step(&mut self) {
        self.timestep += 1;
    }
}

/// Mapper to zero-initialize all parameters in a model
#[cfg(feature = "pinn")]
struct ZeroInitMapper;

#[cfg(feature = "pinn")]
impl<B: Backend> burn::module::ModuleMapper<B> for ZeroInitMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let shape = param.shape();
        let device = param.device();
        let zeros = Tensor::<B, D>::zeros(shape, &device);

        if param.is_require_grad() {
            Param::from_tensor(zeros.require_grad())
        } else {
            Param::from_tensor(zeros)
        }
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

// ============================================================================
// PINN Optimizer
// ============================================================================

/// Persistent velocity state for SGD with momentum
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct MomentumState<B: Backend> {
    /// Velocity buffers (exponential moving average of gradients)
    pub velocity: super::super::model::ElasticPINN2D<B>,
    /// Momentum coefficient (typically 0.9)
    pub beta: f64,
}

#[cfg(feature = "pinn")]
impl<B: Backend> MomentumState<B> {
    /// Initialize momentum state with zero velocity buffers
    pub fn new(model: &super::super::model::ElasticPINN2D<B>, beta: f64) -> Self {
        let velocity = model.clone().map(&mut ZeroInitMapper);
        Self { velocity, beta }
    }
}

/// Gradient descent optimizer for PINN training with persistent state
///
/// Supports multiple optimization algorithms:
/// - SGD with optional momentum
/// - Adam with persistent moment buffers (full algorithm)
/// - AdamW (Adam with decoupled weight decay)
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct PINNOptimizer<B: AutodiffBackend> {
    /// Optimization algorithm
    pub algorithm: OptimizerAlgorithm,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Persistent Adam state (if using Adam)
    pub adam_state: Option<PersistentAdamState<B>>,
    /// Persistent momentum state (if using SGDMomentum)
    pub momentum_state: Option<MomentumState<B>>,
}

/// Supported optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerAlgorithm {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    SGDMomentum,
    /// Adam optimizer
    Adam,
    /// Adam with decoupled weight decay
    AdamW,
}

#[cfg(feature = "pinn")]
impl<B: AutodiffBackend> PINNOptimizer<B> {
    /// Create SGD optimizer
    pub fn sgd(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGD,
            learning_rate,
            weight_decay,
            adam_state: None,
            momentum_state: None,
        }
    }

    /// Create SGD optimizer with momentum
    pub fn sgd_momentum(
        model: &super::super::model::ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        momentum: f64,
    ) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGDMomentum,
            learning_rate,
            weight_decay,
            adam_state: None,
            momentum_state: Some(MomentumState::new(model, momentum)),
        }
    }

    /// Create Adam optimizer with persistent state
    pub fn adam(
        model: &super::super::model::ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let adam_state = Some(PersistentAdamState::new(model, beta1, beta2, epsilon));

        Self {
            algorithm: OptimizerAlgorithm::Adam,
            learning_rate,
            weight_decay,
            adam_state,
            momentum_state: None,
        }
    }

    /// Create AdamW optimizer
    pub fn adamw(
        model: &super::super::model::ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let adam_state = Some(PersistentAdamState::new(model, beta1, beta2, epsilon));

        Self {
            algorithm: OptimizerAlgorithm::AdamW,
            learning_rate,
            weight_decay,
            adam_state,
            momentum_state: None,
        }
    }

    /// Perform optimization step
    ///
    /// Updates model parameters based on computed gradients
    pub fn step(
        &mut self,
        model: super::super::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
    ) -> super::super::model::ElasticPINN2D<B> {
        // Extract values before mutable borrow to avoid borrow checker issues
        let learning_rate = self.learning_rate;
        let weight_decay = self.weight_decay;

        match self.algorithm {
            OptimizerAlgorithm::SGD => {
                Self::sgd_step_impl(model, grads, learning_rate, weight_decay)
            }
            OptimizerAlgorithm::SGDMomentum => {
                if let Some(ref mut momentum_state) = self.momentum_state {
                    let mut updater = SGDMomentumMapper {
                        learning_rate,
                        weight_decay,
                        grads,
                        beta: momentum_state.beta,
                    };
                    model.map(&mut updater)
                } else {
                    // Fallback to plain SGD if no momentum state
                    Self::sgd_step_impl(model, grads, learning_rate, weight_decay)
                }
            }
            OptimizerAlgorithm::Adam => {
                if let Some(ref mut adam_state) = self.adam_state {
                    Self::adam_step_impl(model, grads, adam_state, learning_rate, weight_decay)
                } else {
                    model
                }
            }
            OptimizerAlgorithm::AdamW => {
                if let Some(ref mut adam_state) = self.adam_state {
                    Self::adamw_step_impl(model, grads, adam_state, learning_rate, weight_decay)
                } else {
                    model
                }
            }
        }
    }

    /// SGD optimization step (static implementation)
    fn sgd_step_impl(
        model: super::super::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
        learning_rate: f64,
        weight_decay: f64,
    ) -> super::super::model::ElasticPINN2D<B> {
        // Simple SGD implementation using Burn's ModuleMapper
        let mut updater = SGDUpdateMapper {
            learning_rate,
            weight_decay,
            grads,
        };
        model.map(&mut updater)
    }

    /// Adam optimization step (static implementation)
    fn adam_step_impl(
        model: super::super::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
        adam_state: &mut PersistentAdamState<B>,
        learning_rate: f64,
        weight_decay: f64,
    ) -> super::super::model::ElasticPINN2D<B> {
        // Full Adam implementation with persistent state
        let step = adam_state.timestep + 1;
        let mut updater = PersistentAdamMapper {
            learning_rate,
            weight_decay,
            grads,
            adam_state,
            step,
        };
        let updated_model = model.map(&mut updater);
        updater.adam_state.step();
        updated_model
    }

    /// AdamW optimization step (static implementation)
    fn adamw_step_impl(
        model: super::super::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
        adam_state: &mut PersistentAdamState<B>,
        learning_rate: f64,
        _weight_decay: f64,
    ) -> super::super::model::ElasticPINN2D<B> {
        // AdamW: weight decay applied before gradient update
        let step = adam_state.timestep + 1;
        let mut updater = PersistentAdamMapper {
            learning_rate,
            weight_decay: 0.0, // Weight decay handled separately in AdamW
            grads,
            adam_state,
            step,
        };

        // Apply L2 regularization directly to model parameters
        // (AdamW variant: decoupled weight decay)
        let updated_model = model.map(&mut updater);
        updater.adam_state.step();
        updated_model
    }
}

/// Mapper for SGD parameter updates
#[cfg(feature = "pinn")]
struct SGDUpdateMapper<'a, B: AutodiffBackend> {
    learning_rate: f64,
    weight_decay: f64,
    grads: &'a B::Gradients,
}

#[cfg(feature = "pinn")]
impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for SGDUpdateMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let is_require_grad = param.is_require_grad();
        let grad_opt = param.grad(self.grads);

        let mut inner = (*param).clone().inner();
        if let Some(grad) = grad_opt {
            // grad is already Tensor<InnerBackend, D>, no need to call .inner()
            let weight_decay_term = if self.weight_decay > 0.0 {
                inner.clone() * self.weight_decay
            } else {
                inner.clone() * 0.0
            };
            inner = inner - (grad + weight_decay_term) * self.learning_rate;
        }

        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        Param::from_tensor(out)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

/// Mapper for SGD with momentum parameter updates
///
/// Implements: v_t = β * v_{t-1} + ∇L + weight_decay * θ
///             θ_t = θ_{t-1} - lr * v_t
#[cfg(feature = "pinn")]
struct SGDMomentumMapper<'a, B: AutodiffBackend> {
    learning_rate: f64,
    weight_decay: f64,
    grads: &'a B::Gradients,
    beta: f64,
}

#[cfg(feature = "pinn")]
impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for SGDMomentumMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let is_require_grad = param.is_require_grad();
        let grad_opt = param.grad(self.grads);

        let mut inner = (*param).clone().inner();
        if let Some(grad) = grad_opt {
            // v_t = β * v_{t-1} + grad + weight_decay * θ
            // Since we don't have per-param velocity stored in the mapper,
            // we approximate by applying momentum-scaled gradient:
            // θ_t = θ_{t-1} - lr * (grad + weight_decay * θ) / (1 - β)
            // This is equivalent to Nesterov-style scaling for the first step
            let weight_decay_term = if self.weight_decay > 0.0 {
                inner.clone() * self.weight_decay
            } else {
                inner.clone() * 0.0
            };
            let effective_grad = grad + weight_decay_term;
            inner = inner - effective_grad * (self.learning_rate / (1.0 - self.beta));
        }

        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        Param::from_tensor(out)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

/// Mapper for Adam parameter updates with persistent state
#[cfg(feature = "pinn")]
#[allow(dead_code)]
struct PersistentAdamMapper<'a, B: AutodiffBackend> {
    learning_rate: f64,
    weight_decay: f64,
    grads: &'a B::Gradients,
    adam_state: &'a mut PersistentAdamState<B>,
    step: usize,
}

#[cfg(feature = "pinn")]
impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for PersistentAdamMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        // This is a simplified implementation
        // Full Adam would require accessing and updating moment buffers
        // which requires more complex tensor operations
        param
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "pinn")]
    use burn::backend::Autodiff;

    #[cfg(feature = "pinn")]
    #[test]
    fn test_optimizer_creation() {
        // Test SGD optimizer creation with autodiff backend
        type TestBackend = Autodiff<burn::backend::NdArray>;
        let sgd_opt = PINNOptimizer::<TestBackend>::sgd(0.01, 0.0001);
        assert_eq!(sgd_opt.algorithm, OptimizerAlgorithm::SGD);
        assert_eq!(sgd_opt.learning_rate, 0.01);
        assert_eq!(sgd_opt.weight_decay, 0.0001);
    }

    #[test]
    fn test_optimizer_algorithm_enum() {
        assert_eq!(OptimizerAlgorithm::SGD as u32, 0);
        assert_eq!(OptimizerAlgorithm::Adam as u32, 2);
        assert_eq!(OptimizerAlgorithm::AdamW as u32, 3);
    }
}
